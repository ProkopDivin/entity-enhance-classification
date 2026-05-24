"""Build a LaTeX comparison table from ``evaluation_comparison`` outputs.

For each subdirectory under the input comparison root the script extracts:

- Test F1 / Precision / Recall from ``summary_comparison.csv`` (row
  ``micro_over_labels``). The shared baseline row reuses the ``*_base``
  columns from any subdirectory.
- CV F1 mean / std from the ``dev_cv_summary`` sheet of the latest matching
  ``final_evaluation_tables_*.xlsx`` in ``results/saved_models``.

If the root contains ``summary_comparison.csv`` but there are no experiment
subdirectories with resolvable names, the root directory itself is treated as
a single experiment (same CSV layout as a leaf folder). The saved-model
``config_name`` is taken from the folder name (``_vs_`` prefix or YAML
``overrides``) or from ``--flat-config-name`` when the folder name does not
encode the config (e.g. timestamped bundle dirs).

Display tags are derived automatically from the part of the directory name
before ``_vs_`` (e.g. ``wpentities_weighted_mean_vs_best_article_only_...`` ->
``wpentities_weighted_mean``) and then translated through the ``aliases``
substitutions defined in ``resources/latex_table_mapping.yaml`` (e.g.
``wpentities`` -> ``wpe`` -> ``wpe_weighted_mean``). Legacy directories without
``_vs_`` (the ``_old_analysis`` set) are matched against the YAML
``overrides`` list, which only specifies the underlying config name.

Per-column bolding is recomputed dynamically from the resolved values rounded
to the same precision they are printed at; in case of ties all maximum cells
are bolded.

Example::

    python -m iptc_entity_pipeline.build_latex_table results/comparisons

Flat bundle (CSV files next to the path you pass)::

    python -m iptc_entity_pipeline.build_latex_table results/comparisons/run123 \\
        --flat-config-name best_wpentities_all_langs
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

LOG = logging.getLogger(__name__)

DEFAULT_MAPPING_PATH = Path(__file__).parent / 'resources' / 'latex_table_mapping.yaml'
DEFAULT_SAVED_MODELS_DIR = Path('results/saved_models')
DEFAULT_OUTPUT_SUBDIR = 'latex-table'
SUMMARY_FILENAME = 'summary_comparison.csv'
CORPORA_FILENAME = 'corpora_comparison.csv'
TOP_IMPROVED_STATS_FILENAME = 'top_improved_stats.csv'
TOP_DEGRADED_STATS_FILENAME = 'top_degraded_stats.csv'
SUMMARY_KEY = 'micro_over_labels'
DEV_CV_SHEET = 'dev_cv_summary'
VS_SEPARATOR = '_vs_'
SAVED_MODEL_SUFFIX_RE = re.compile(r'_\d{8}_\d{6}$')
CV_F1_DECIMALS = 4
TEST_DECIMALS = 3
DELTA_DECIMALS = 2
CURRENT_DECIMALS = 2


@dataclass(frozen=True)
class Override:
    """One legacy comparison-dir prefix -> underlying config name."""

    prefix: str
    config_name: str


@dataclass(frozen=True)
class Baseline:
    """Baseline row metadata."""

    display_tag: str
    config_name: str


@dataclass(frozen=True)
class MappingConfig:
    """Resolved mapping configuration."""

    aliases: tuple[tuple[str, str], ...]
    baseline: Baseline
    overrides: tuple[Override, ...]
    display_order: tuple[str, ...]


@dataclass(frozen=True)
class TableRow:
    """One LaTeX row with all metrics already coerced to floats."""

    tag: str
    cv_f1: float | None
    cv_f1_std: float | None
    test_f1: float | None
    test_prec: float | None
    test_rec: float | None


@dataclass(frozen=True)
class ExperimentEntry:
    """Metadata for one experiment comparison subdirectory."""

    subdir: Path
    config_name: str
    display_tag: str


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_mapping(*, mapping_path: Path) -> MappingConfig:
    """Parse the YAML mapping file."""
    with open(mapping_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    aliases = tuple((str(k), str(v)) for k, v in (raw.get('aliases') or {}).items())
    baseline_raw = raw.get('baseline') or {}
    baseline = Baseline(
        display_tag=str(baseline_raw.get('display_tag', 'baseline')),
        config_name=str(baseline_raw.get('config_name', '')),
    )
    overrides = tuple(
        sorted(
            (
                Override(prefix=str(item['prefix']), config_name=str(item['config_name']))
                for item in raw.get('overrides') or []
            ),
            key=lambda o: len(o.prefix),
            reverse=True,
        )
    )
    display_order = tuple(str(tag) for tag in raw.get('display_order') or [])
    return MappingConfig(
        aliases=aliases,
        baseline=baseline,
        overrides=overrides,
        display_order=display_order,
    )


def apply_aliases(*, name: str, aliases: Sequence[tuple[str, str]]) -> str:
    """Apply substring substitutions in declaration order."""
    result = name
    for src, dst in aliases:
        if src:
            result = result.replace(src, dst)
    return result.strip('_')


def resolve_config_name(*, dir_name: str, overrides: Sequence[Override]) -> str | None:
    """Derive the saved-model config name from a comparison directory name.

    Auto-derivation: take the part before the first ``_vs_`` separator.
    Fallback: longest-prefix match in ``overrides``.
    Returns ``None`` when neither method yields a result.
    """
    if VS_SEPARATOR in dir_name:
        return dir_name.split(VS_SEPARATOR, 1)[0]
    for override in overrides:
        if dir_name.startswith(override.prefix):
            return override.config_name
    return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def read_test_metrics(*, summary_csv: Path) -> Mapping[str, float] | None:
    """Read the ``micro_over_labels`` row from a comparison summary CSV."""
    df = pd.read_csv(summary_csv)
    rows = df[df['summary_key'] == SUMMARY_KEY]
    if rows.empty:
        LOG.warning(f'Missing summary_key={SUMMARY_KEY} in {summary_csv}')
        return None
    row = rows.iloc[0]
    return {
        'precision_current': float(row['precision_current']),
        'recall_current': float(row['recall_current']),
        'f1_current': float(row['f1_current']),
        'precision_base': float(row['precision_base']),
        'recall_base': float(row['recall_base']),
        'f1_base': float(row['f1_base']),
    }


def find_saved_model_dir(*, saved_models_dir: Path, config_name: str) -> Path | None:
    """Return the latest ``<config_name>_<TS>`` directory.

    Falls back to ``best_<config_name>_<TS>`` when no exact match exists, so
    that auto-derived config names like ``wpentities_all_langs`` still resolve
    against directories such as ``best_wpentities_all_langs_<TS>``.
    """
    if not config_name or not saved_models_dir.is_dir():
        return None
    candidates = [config_name]
    if not config_name.startswith('best_'):
        candidates.append(f'best_{config_name}')
    for candidate in candidates:
        prefix = f'{candidate}_'
        matches = [
            child
            for child in saved_models_dir.iterdir()
            if child.is_dir()
            and child.name.startswith(prefix)
            and SAVED_MODEL_SUFFIX_RE.search(child.name[len(candidate):])
            and child.name[len(candidate):].count('_') == 2
        ]
        if matches:
            return sorted(matches, key=lambda p: p.name)[-1]
    return None


def read_cv_metrics(*, model_dir: Path) -> tuple[float, float] | None:
    """Read ``F1_micro`` / ``F1_micro_std`` from the ``dev_cv_summary`` sheet of the model dir."""
    xlsx_files = sorted(model_dir.glob('final_evaluation_tables_*.xlsx'))
    if not xlsx_files:
        LOG.warning(f'No final_evaluation_tables_*.xlsx in {model_dir}')
        return None
    df = pd.read_excel(xlsx_files[0], sheet_name=DEV_CV_SHEET)
    if df.empty:
        LOG.warning(f'Empty {DEV_CV_SHEET} sheet in {xlsx_files[0]}')
        return None
    row = df.iloc[0]
    return float(row['F1_micro']), float(row['F1_micro_std'])


def fetch_cv_metrics(
    *,
    saved_models_dir: Path,
    config_name: str,
) -> tuple[float, float] | None:
    """Locate the saved-model dir for ``config_name`` and load CV metrics."""
    model_dir = find_saved_model_dir(saved_models_dir=saved_models_dir, config_name=config_name)
    if model_dir is None:
        LOG.warning(f'No saved-model directory found for config_name={config_name}')
        return None
    return read_cv_metrics(model_dir=model_dir)


# ---------------------------------------------------------------------------
# Row collection
# ---------------------------------------------------------------------------

SKIP_DIR_NAMES = frozenset({'old-comparisons', DEFAULT_OUTPUT_SUBDIR})


def list_comparison_subdirs(*, comparison_dir: Path) -> list[Path]:
    """List comparison subdirectories, skipping leading-underscore folders."""
    subdirs: list[Path] = []
    for child in sorted(comparison_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith('_') or child.name in SKIP_DIR_NAMES:
            continue
        subdirs.append(child)
    return subdirs


def collect_experiments(
    *,
    comparison_dir: Path,
    config: MappingConfig,
    flat_config_name: str | None = None,
) -> list[ExperimentEntry]:
    """Collect experiment metadata for all valid comparison subdirectories.

    When no subdirectory yields an experiment but ``summary_comparison.csv``
    exists on ``comparison_dir`` itself, the root is treated as one experiment
    (flat bundle layout).
    """
    experiments: list[ExperimentEntry] = []
    for sub in list_comparison_subdirs(comparison_dir=comparison_dir):
        config_name = resolve_config_name(dir_name=sub.name, overrides=config.overrides)
        if config_name is None:
            LOG.warning(f'Could not resolve config_name for {sub.name}; skipping')
            continue
        display_tag = apply_aliases(name=config_name, aliases=config.aliases)
        experiments.append(
            ExperimentEntry(
                subdir=sub,
                config_name=config_name,
                display_tag=display_tag,
            )
        )

    if experiments:
        return experiments

    summary_at_root = comparison_dir / SUMMARY_FILENAME
    if not summary_at_root.is_file():
        return experiments

    if flat_config_name and flat_config_name.strip():
        cn = flat_config_name.strip()
    else:
        cn = resolve_config_name(dir_name=comparison_dir.name, overrides=config.overrides)
    if cn is None:
        LOG.warning(
            f'Found {SUMMARY_FILENAME} at comparison root {comparison_dir} but could not '
            f'derive config_name from directory name {comparison_dir.name!r}. '
            'Use --flat-config-name (saved-model prefix, e.g. best_wpentities_all_langs) '
            'or rename the directory to <config>_vs_<baseline>_...'
        )
        return experiments

    display_tag = apply_aliases(name=cn, aliases=config.aliases)
    experiments.append(
        ExperimentEntry(
            subdir=comparison_dir,
            config_name=cn,
            display_tag=display_tag,
        )
    )
    LOG.info(
        f'Using flat comparison layout at {comparison_dir} '
        f'(config_name={cn}, display_tag={display_tag})'
    )
    return experiments


def collect_rows(
    *,
    comparison_dir: Path,
    saved_models_dir: Path,
    config: MappingConfig,
    flat_config_name: str | None = None,
) -> list[TableRow]:
    """Walk subdirectories and assemble one row per matched experiment."""
    experiment_rows: dict[str, TableRow] = {}
    base_metrics: Mapping[str, float] | None = None

    for experiment in collect_experiments(
        comparison_dir=comparison_dir,
        config=config,
        flat_config_name=flat_config_name,
    ):
        summary_csv = experiment.subdir / SUMMARY_FILENAME
        if not summary_csv.is_file():
            LOG.warning(f'Skipping {experiment.subdir.name}: missing {SUMMARY_FILENAME}')
            continue
        metrics = read_test_metrics(summary_csv=summary_csv)
        if metrics is None:
            continue
        if base_metrics is None:
            base_metrics = metrics
        cv = fetch_cv_metrics(saved_models_dir=saved_models_dir, config_name=experiment.config_name)
        row = TableRow(
            tag=experiment.display_tag,
            cv_f1=cv[0] if cv else None,
            cv_f1_std=cv[1] if cv else None,
            test_f1=metrics['f1_current'],
            test_prec=metrics['precision_current'],
            test_rec=metrics['recall_current'],
        )
        if experiment.display_tag in experiment_rows:
            LOG.info(
                f'Duplicate tag {experiment.display_tag}; '
                f'keeping most recent subdir {experiment.subdir.name}'
            )
        experiment_rows[experiment.display_tag] = row

    ordered_rows = order_rows(rows=experiment_rows, display_order=config.display_order)

    if base_metrics is None:
        LOG.error('No comparison subdirectory contained the required summary row.')
        return ordered_rows

    cv_base = fetch_cv_metrics(
        saved_models_dir=saved_models_dir,
        config_name=config.baseline.config_name,
    )
    baseline_row = TableRow(
        tag=config.baseline.display_tag,
        cv_f1=cv_base[0] if cv_base else None,
        cv_f1_std=cv_base[1] if cv_base else None,
        test_f1=base_metrics['f1_base'],
        test_prec=base_metrics['precision_base'],
        test_rec=base_metrics['recall_base'],
    )
    return [baseline_row, *ordered_rows]


def order_rows(
    *,
    rows: Mapping[str, TableRow],
    display_order: Sequence[str],
) -> list[TableRow]:
    """Order rows by ``display_order`` first, then alphabetically."""
    ordered: list[TableRow] = []
    seen: set[str] = set()
    for tag in display_order:
        if tag in rows:
            ordered.append(rows[tag])
            seen.add(tag)
    for tag in sorted(rows):
        if tag not in seen:
            ordered.append(rows[tag])
    return ordered


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def find_max_indices(*, values: Sequence[float | None], decimals: int) -> set[int]:
    """Return indices of the maximum value rounded to ``decimals`` places.

    Comparison happens on the displayed (rounded) value so that ties visible in
    the output table all receive the bold treatment.
    """
    valid = [(idx, round(value, decimals)) for idx, value in enumerate(values) if value is not None]
    if not valid:
        return set()
    max_val = max(value for _, value in valid)
    return {idx for idx, value in valid if value == max_val}


def fmt_value(*, value: float | None, decimals: int, bold: bool) -> str:
    """Format ``value`` to ``decimals`` and optionally wrap in ``\\textbf{}``."""
    if value is None:
        return '--'
    text = format(value, f'.{decimals}f')
    return f'\\textbf{{{text}}}' if bold else text


def escape_tag(*, tag: str) -> str:
    """LaTeX-escape underscores in a display tag."""
    return tag.replace('_', '\\_')


def render_latex(*, rows: Sequence[TableRow], label: str, caption: str) -> str:
    """Render the assembled rows as a resizable LaTeX table."""
    cv_f1_max = find_max_indices(values=[r.cv_f1 for r in rows], decimals=CV_F1_DECIMALS)
    test_f1_max = find_max_indices(values=[r.test_f1 for r in rows], decimals=TEST_DECIMALS)
    test_prec_max = find_max_indices(values=[r.test_prec for r in rows], decimals=TEST_DECIMALS)
    test_rec_max = find_max_indices(values=[r.test_rec for r in rows], decimals=TEST_DECIMALS)

    lines: list[str] = [
        '\\begin{table}[h!]',
        '\\centering',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lccccc}',
        '\\hline',
        'Tag & CV F1 & CV F1 Std & Test F1 & Test Prec & Test Rec \\\\',
        '\\hline',
    ]
    for idx, row in enumerate(rows):
        cells = (
            escape_tag(tag=row.tag),
            fmt_value(value=row.cv_f1, decimals=CV_F1_DECIMALS, bold=idx in cv_f1_max),
            fmt_value(value=row.cv_f1_std, decimals=CV_F1_DECIMALS, bold=False),
            fmt_value(value=row.test_f1, decimals=TEST_DECIMALS, bold=idx in test_f1_max),
            fmt_value(value=row.test_prec, decimals=TEST_DECIMALS, bold=idx in test_prec_max),
            fmt_value(value=row.test_rec, decimals=TEST_DECIMALS, bold=idx in test_rec_max),
        )
        lines.append(' & '.join(cells) + ' \\\\')
    lines.extend(
        [
            '\\hline',
            '\\end{tabular}%',
            '}',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{table}',
        ]
    )
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Per-experiment LaTeX rendering
# ---------------------------------------------------------------------------

def as_float(*, value: Any) -> float:
    """Best-effort conversion to float with NaN fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def fmt_plain(*, value: float, decimals: int = CURRENT_DECIMALS) -> str:
    """Format numeric value with fixed decimals."""
    if abs(value) < (0.5 * (10 ** -decimals)):
        value = 0.0
    return format(value, f'.{decimals}f')


def fmt_delta(*, value: float, bold: bool, decimals: int = DELTA_DECIMALS) -> str:
    """Format delta value and optionally apply bold style."""
    if abs(value) < (0.5 * (10 ** -decimals)):
        value = 0.0
    text = format(value, f'.{decimals}f')
    return f'\\textbf{{{text}}}' if bold else text


def sanitize_token(*, text: str) -> str:
    """Build ASCII-safe token for filenames / labels."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]+', '_', text)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned.lower() or 'table'


def top_k_indices(*, values: Sequence[float], labels: Sequence[str], k: int = 3) -> set[int]:
    """Return indices in the top-k values (including ties at cutoff)."""
    valid_indices = [idx for idx, value in enumerate(values) if pd.notna(value)]
    if not valid_indices:
        return set()
    ranked = sorted(
        [(idx, values[idx], labels[idx]) for idx in valid_indices],
        key=lambda item: (-item[1], item[2]),
    )
    cutoff_pos = min(k, len(ranked)) - 1
    cutoff_value = ranked[cutoff_pos][1]
    return {idx for idx, value, _ in ranked if value >= cutoff_value}


def render_experiment_corpora_table(*, experiment: ExperimentEntry, corpora_df: pd.DataFrame) -> str:
    """Render corpus-level table for one experiment."""
    df = corpora_df.copy()
    f1_deltas = [as_float(value=value) for value in df['F1_diff']]
    corpus_labels = [str(value) for value in df['Corpus Name']]
    top_delta_indices = top_k_indices(values=f1_deltas, labels=corpus_labels, k=3)
    lines: list[str] = [
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrrrrrr}',
        '\\toprule',
        '\\multirow{2}{*}{\\textbf{Corpus}}',
        '& \\multicolumn{2}{c}{\\textbf{Precision}}',
        '& \\multicolumn{2}{c}{\\textbf{Recall}}',
        '& \\multicolumn{2}{c}{\\textbf{F1 Score}} \\\\',
        '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base} \\\\',
        '\\midrule',
    ]
    for idx, (_, row) in enumerate(df.iterrows()):
        f1_diff = as_float(value=row['F1_diff'])
        cells = [
            escape_tag(tag=str(row['Corpus Name'])),
            fmt_plain(value=as_float(value=row['Precision_current'])),
            fmt_delta(value=as_float(value=row['Precision_diff']), bold=False),
            fmt_plain(value=as_float(value=row['Recall_current'])),
            fmt_delta(value=as_float(value=row['Recall_diff']), bold=False),
            fmt_plain(value=as_float(value=row['F1_current'])),
            fmt_delta(value=f1_diff, bold=idx in top_delta_indices),
        ]
        lines.append(' & '.join(cells) + ' \\\\')
    caption = (
        f'Results for \\textit{{{escape_tag(tag=experiment.display_tag)}}} model compared to the base model across '
        'corpora. For each metric, the table reports the current score and its difference relative to the base model. '
        'The top 3 values in the F1 Score $\\Delta$ column are highlighted in bold.'
    )
    label = f'tab:{sanitize_token(text=experiment.display_tag)}-vs-base-corpora'
    lines.extend(
        [
            '\\bottomrule',
            '\\end{tabular}%',
            '}',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{table*}',
            '',
        ]
    )
    return '\n'.join(lines)


SUMMARY_GROUP_ROWS: tuple[tuple[str, str, str], ...] = (
    ('support', 'Support 0--10', 'macro_over_classes_support_0-10'),
    ('support', 'Support 10--100', 'macro_over_classes_support_10-100'),
    ('support', 'Support 100--1000', 'macro_over_classes_support_100-1000'),
    ('support', 'Support 1000--10000', 'macro_over_classes_support_1000-10000'),
    ('lang', 'EN', 'macro_over_corpora_prefix_en'),
    ('lang', 'ES', 'macro_over_corpora_prefix_es'),
    ('lang', 'NL', 'macro_over_corpora_prefix_nl'),
    ('lang', 'FR', 'macro_over_corpora_prefix_fr'),
    ('lang', 'DE', 'macro_over_corpora_prefix_de'),
    ('lang', 'CS', 'macro_over_corpora_prefix_cs'),
    ('corpora', 'Eurosport', 'macro_over_corpora_eurosport'),
    ('corpora', 'All corpora', 'macro_over_corpora'),
)


def render_experiment_group_summary_table(*, experiment: ExperimentEntry, summary_df: pd.DataFrame) -> str:
    """Render grouped summary table for one experiment."""
    key_to_row = {str(row.summary_key): row for row in summary_df.itertuples(index=False)}
    rows_data: list[tuple[str, str, float, float, float, float, float, float]] = []
    for section, label, key in SUMMARY_GROUP_ROWS:
        row = key_to_row.get(key)
        if row is None:
            LOG.warning(f'Missing summary row key={key} in {experiment.subdir / SUMMARY_FILENAME}')
            continue
        rows_data.append(
            (
                section,
                label,
                as_float(value=row.precision_current),
                as_float(value=row.precision_diff),
                as_float(value=row.recall_current),
                as_float(value=row.recall_diff),
                as_float(value=row.f1_current),
                as_float(value=row.f1_diff),
            )
        )
    f1_deltas = [item[7] for item in rows_data]
    labels = [item[1] for item in rows_data]
    top_delta_indices = top_k_indices(values=f1_deltas, labels=labels, k=3) if rows_data else set()
    section_titles = {
        'support': '\\multicolumn{7}{l}{\\textbf{Support buckets}} \\\\',
        'lang': '\\multicolumn{7}{l}{\\textbf{Language prefixes}} \\\\',
        'corpora': '\\multicolumn{7}{l}{\\textbf{Corpora}} \\\\',
    }
    lines: list[str] = [
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrrrrrr}',
        '\\toprule',
        '\\multirow{2}{*}{\\textbf{Group}}',
        '& \\multicolumn{2}{c}{\\textbf{Precision}}',
        '& \\multicolumn{2}{c}{\\textbf{Recall}}',
        '& \\multicolumn{2}{c}{\\textbf{F1 Score}} \\\\',
        '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base}',
        '& \\textbf{Current} & \\textbf{$\\Delta$ vs Base} \\\\',
        '\\midrule',
    ]
    current_section = ''
    for idx, item in enumerate(rows_data):
        section, label, p_cur, p_diff, r_cur, r_diff, f1_cur, f1_diff = item
        if current_section != section:
            if current_section:
                lines.append('')
                lines.append('\\midrule')
            lines.append(section_titles[section])
            current_section = section
        lines.append(
            ' & '.join(
                [
                    escape_tag(tag=label),
                    fmt_plain(value=p_cur),
                    fmt_delta(value=p_diff, bold=False),
                    fmt_plain(value=r_cur),
                    fmt_delta(value=r_diff, bold=False),
                    fmt_plain(value=f1_cur),
                    fmt_delta(value=f1_diff, bold=idx in top_delta_indices),
                ]
            )
            + ' \\\\'
        )
    caption = (
        f'Results for \\textit{{{escape_tag(tag=experiment.display_tag)}}} compared to the base model across '
        'aggregated groups. The table reports current scores and their differences relative to the base model. '
        'The top 3 values in the F1 Score $\\Delta$ column are highlighted in bold.'
    )
    label = f'tab:{sanitize_token(text=experiment.display_tag)}-summary'
    lines.extend(
        [
            '',
            '\\bottomrule',
            '\\end{tabular}%',
            '}',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{table*}',
            '',
        ]
    )
    return '\n'.join(lines)


def read_stats_csv(*, path: Path) -> dict[str, float]:
    """Read metric-value stats CSV into a dict."""
    df = pd.read_csv(path)
    result: dict[str, float] = {}
    for row in df.itertuples(index=False):
        result[str(row.metric)] = as_float(value=row.value)
    return result


def title_case_category(*, name: str) -> str:
    """Convert lower-case top-level category label to display form."""
    text = name.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def render_improvement_degradation_table(
    *,
    experiment: ExperimentEntry,
    improved_stats: Mapping[str, float],
    degraded_stats: Mapping[str, float],
) -> str:
    """Render improvement vs degradation table from stats files."""
    summary_rows: list[tuple[str, str, str]] = [
        (
            'Number of categories',
            str(int(improved_stats.get('count_improved', 0))),
            str(int(degraded_stats.get('count_degraded', 0))),
        ),
        (
            'Categories with $\\Delta$F1 $> 0.1$ / $< -0.1$',
            str(int(improved_stats.get('count_improved_f1_diff_gt_0.1', 0))),
            str(int(degraded_stats.get('count_degraded_f1_diff_gt_0.1', 0))),
        ),
        (
            'Categories with $|\\Delta$F1$| > 0.3$',
            str(int(improved_stats.get('count_improved_f1_diff_gt_0.3', 0))),
            str(int(degraded_stats.get('count_degraded_f1_diff_gt_0.3', 0))),
        ),
        (
            'Categories with $|\\Delta$F1$| > 0.5$',
            str(int(improved_stats.get('count_improved_f1_diff_gt_0.5', 0))),
            str(int(degraded_stats.get('count_degraded_f1_diff_gt_0.5', 0))),
        ),
        (
            'Categories with $|\\Delta$F1$| > 0.7$',
            str(int(improved_stats.get('count_improved_f1_diff_gt_0.7', 0))),
            str(int(degraded_stats.get('count_degraded_f1_diff_gt_0.7', 0))),
        ),
        (
            'Average $\\Delta$F1 (top 100)',
            format(as_float(value=improved_stats.get('avg_f1_diff_top_100', 0.0)), '.4f'),
            format(as_float(value=degraded_stats.get('avg_f1_diff_top_100', 0.0)), '.4f'),
        ),
        (
            'Average article frequency (top 100)',
            format(as_float(value=improved_stats.get('avg_article_frequency_top_100', 0.0)), '.2f'),
            format(as_float(value=degraded_stats.get('avg_article_frequency_top_100', 0.0)), '.2f'),
        ),
    ]
    prefix = 'top_level_top_100::'
    improved_levels = {k.removeprefix(prefix): v for k, v in improved_stats.items() if k.startswith(prefix)}
    degraded_levels = {k.removeprefix(prefix): v for k, v in degraded_stats.items() if k.startswith(prefix)}
    all_level_names = sorted(set(improved_levels) | set(degraded_levels))
    level_rows = sorted(
        [
            (
                title_case_category(name=name),
                int(as_float(value=improved_levels.get(name, 0.0))),
                int(as_float(value=degraded_levels.get(name, 0.0))),
            )
            for name in all_level_names
        ],
        key=lambda item: (-(item[1] + item[2]), item[0]),
    )
    lines: list[str] = [
        '\\begin{table}[t]',
        '\\centering',
        '\\small',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrr}',
        '\\toprule',
        '\\textbf{Metric} & \\textbf{Improvement} & \\textbf{Degradation} \\\\',
        '\\midrule',
        '',
        '\\multicolumn{3}{l}{\\textbf{Summary}} \\\\',
    ]
    for metric, improved, degraded in summary_rows:
        lines.append(f'{metric} & {improved} & {degraded} \\\\')
    lines.extend(
        [
            '',
            '\\midrule',
            '\\multicolumn{3}{l}{\\textbf{Top-level category composition (top 100)}} \\\\',
        ]
    )
    for name, improved_count, degraded_count in level_rows:
        lines.append(f'{name} & {improved_count} & {degraded_count} \\\\')
    caption = (
        'Comparison of categories with the largest improvements and degradations relative to the base model. '
        'The table reports the number of affected categories, thresholded counts based on F1 change, average F1 '
        'improvement and article frequency among the top 100 categories, and the distribution of top-level '
        'categories within the top 100 improved and degraded groups.'
    )
    label = f'tab:{sanitize_token(text=experiment.display_tag)}-improvement-vs-degradation'
    lines.extend(
        [
            '',
            '\\bottomrule',
            '\\end{tabular}%',
            '}',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{table}',
            '',
        ]
    )
    return '\n'.join(lines)


def write_per_experiment_tables(*, experiments: Sequence[ExperimentEntry], output_dir: Path) -> int:
    """Generate and write 3 LaTeX tables for each experiment."""
    written = 0
    for experiment in experiments:
        corpora_path = experiment.subdir / CORPORA_FILENAME
        summary_path = experiment.subdir / SUMMARY_FILENAME
        improved_path = experiment.subdir / TOP_IMPROVED_STATS_FILENAME
        degraded_path = experiment.subdir / TOP_DEGRADED_STATS_FILENAME
        required_paths = (corpora_path, summary_path, improved_path, degraded_path)
        if not all(path.is_file() for path in required_paths):
            LOG.warning(
                f'Skipping per-experiment tables for {experiment.subdir.name}: '
                'required CSV files are missing.'
            )
            continue
        corpora_df = pd.read_csv(corpora_path)
        summary_df = pd.read_csv(summary_path)
        improved_stats = read_stats_csv(path=improved_path)
        degraded_stats = read_stats_csv(path=degraded_path)
        file_stem = sanitize_token(text=experiment.subdir.name)
        corpora_tex = render_experiment_corpora_table(experiment=experiment, corpora_df=corpora_df)
        summary_tex = render_experiment_group_summary_table(experiment=experiment, summary_df=summary_df)
        imp_deg_tex = render_improvement_degradation_table(
            experiment=experiment,
            improved_stats=improved_stats,
            degraded_stats=degraded_stats,
        )
        (output_dir / f'{file_stem}_corpora.tex').write_text(corpora_tex, encoding='utf-8')
        (output_dir / f'{file_stem}_summary_groups.tex').write_text(summary_tex, encoding='utf-8')
        (output_dir / f'{file_stem}_improvement_vs_degradation.tex').write_text(imp_deg_tex, encoding='utf-8')
        written += 3
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def derive_output_name(*, comparison_dir: Path) -> str:
    """Derive a clean filename stem from the comparison directory name."""
    cleaned = comparison_dir.name.lstrip('_').rstrip('_')
    return cleaned or 'comparison_table'


def main() -> None:
    """Entry point: read comparison + saved-model dirs, write LaTeX table."""
    parser = argparse.ArgumentParser(
        description='Build a LaTeX comparison table from evaluation_comparison outputs.',
    )
    parser.add_argument(
        'comparison_dir',
        type=Path,
        help='Directory containing per-experiment evaluation_comparison subdirectories.',
    )
    parser.add_argument(
        '--saved-models-dir',
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help=f'Saved models root (default: {DEFAULT_SAVED_MODELS_DIR}).',
    )
    parser.add_argument(
        '--mapping',
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help='YAML mapping file with aliases / overrides / display_order.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: <comparison-dir>/latex-table).',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Overview output filename stem (default: derived from comparison-dir name).',
    )
    parser.add_argument(
        '--label',
        type=str,
        default='tab:wpentities_results',
        help='LaTeX label for the produced table.',
    )
    parser.add_argument(
        '--caption',
        type=str,
        default='Model performance metrics from experiment screen',
        help='LaTeX caption for the produced table.',
    )
    parser.add_argument(
        '--mode',
        choices=('overview', 'per_experiment', 'all'),
        default='all',
        help='Generate only overview table, only per-experiment tables, or both (default: all).',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        help='Logging verbosity.',
    )
    parser.add_argument(
        '--flat-config-name',
        type=str,
        default=None,
        help=(
            'When all comparison CSVs live at the comparison root (no experiment '
            'subfolders), set the saved-model config name if the folder name does '
            'not match <config>_vs_... or YAML overrides (e.g. best_wpentities_all_langs).'
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    comparison_dir = args.comparison_dir.resolve()
    if not comparison_dir.is_dir():
        raise SystemExit(f'Comparison directory not found: {comparison_dir}')

    config = load_mapping(mapping_path=args.mapping)
    output_dir = args.output_dir or (comparison_dir / DEFAULT_OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrote_any = False

    if args.mode in ('overview', 'all'):
        rows = collect_rows(
            comparison_dir=comparison_dir,
            saved_models_dir=args.saved_models_dir,
            config=config,
            flat_config_name=args.flat_config_name,
        )
        if rows:
            name = args.name or derive_output_name(comparison_dir=comparison_dir)
            output_path = output_dir / f'{name}.tex'
            latex = render_latex(rows=rows, label=args.label, caption=args.caption)
            output_path.write_text(latex, encoding='utf-8')
            LOG.info(f'Wrote LaTeX overview table with {len(rows)} rows to {output_path}')
            wrote_any = True
        else:
            LOG.warning('No rows produced for overview table.')

    if args.mode in ('per_experiment', 'all'):
        experiments = collect_experiments(
            comparison_dir=comparison_dir,
            config=config,
            flat_config_name=args.flat_config_name,
        )
        written = write_per_experiment_tables(experiments=experiments, output_dir=output_dir)
        LOG.info(f'Wrote {written} per-experiment table files into {output_dir}')
        wrote_any = wrote_any or written > 0

    if not wrote_any:
        raise SystemExit('No tables generated; check inputs and mode.')


if __name__ == '__main__':
    main()
