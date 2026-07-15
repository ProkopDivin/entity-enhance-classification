"""Build a LaTeX comparison table from ``evaluation_comparison`` outputs.

For each subdirectory under the input comparison root the script extracts:

- Test F1 / Precision / Recall from ``summary_comparison.csv`` (row
  ``micro``). The shared baseline row reuses the ``*_base``
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
TOP_IMPROVED_FILENAME = 'top_improved.csv'
TOP_DEGRADED_FILENAME = 'top_degraded.csv'
SUMMARY_KEY = 'micro'
DEV_CV_SHEET = 'dev_cv_summary'
VS_SEPARATOR = '_vs_'
SAVED_MODEL_SUFFIX_RE = re.compile(r'_\d{8}_\d{6}$')
PERCENT_SCALE = 100.0
METRIC_DECIMALS = 1


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
    """Read the ``micro`` row from a comparison summary CSV."""
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
    df = pd.read_excel(xlsx_files[-1], sheet_name=DEV_CV_SHEET)
    if df.empty:
        LOG.warning(f'Empty {DEV_CV_SHEET} sheet in {xlsx_files[-1]}')
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
        cn = comparison_dir.name
        LOG.info(
            f'Using directory name {cn!r} as config_name for flat bundle at {comparison_dir}. '
            'Pass --flat-config-name for saved-model CV metric lookup.'
        )

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

def to_pct(*, value: float) -> float:
    """Convert a unit-interval metric to a percentage."""
    return value * PERCENT_SCALE


def find_max_indices(*, values: Sequence[float | None], decimals: int) -> set[int]:
    """Return indices of the maximum value rounded to ``decimals`` places.

    Comparison happens on the displayed (rounded) percentage so that ties visible
    in the output table all receive the bold treatment.
    """
    valid = [
        (idx, round(to_pct(value=value), decimals))
        for idx, value in enumerate(values)
        if value is not None and not pd.isna(value)
    ]
    if not valid:
        return set()
    max_val = max(value for _, value in valid)
    return {idx for idx, value in valid if value == max_val}


def fmt_value(*, value: float | None, decimals: int = METRIC_DECIMALS, bold: bool) -> str:
    """Format a unit-interval metric as a percentage."""
    if value is None or pd.isna(value):
        return '--'
    return fmt_number(value=to_pct(value=value), decimals=decimals, bold=bold)


def escape_tag(*, tag: str) -> str:
    """LaTeX-escape underscores in a display tag."""
    return tag.replace('_', '\\_')


def render_latex(*, rows: Sequence[TableRow], label: str, caption: str) -> str:
    """Render the assembled rows as a resizable LaTeX table."""
    cv_f1_max = find_max_indices(values=[r.cv_f1 for r in rows], decimals=METRIC_DECIMALS)
    test_f1_max = find_max_indices(values=[r.test_f1 for r in rows], decimals=METRIC_DECIMALS)
    test_prec_max = find_max_indices(values=[r.test_prec for r in rows], decimals=METRIC_DECIMALS)
    test_rec_max = find_max_indices(values=[r.test_rec for r in rows], decimals=METRIC_DECIMALS)

    lines: list[str] = [
        '\\begin{table}[h!]',
        '\\centering',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lccccc}',
        '\\hline',
        'Tag & CV F1 (\\%) & CV F1 Std (\\%) & Test F1 (\\%) & Test Prec (\\%) & Test Rec (\\%) \\\\',
        '\\hline',
    ]
    for idx, row in enumerate(rows):
        cells = (
            escape_tag(tag=row.tag),
            fmt_value(value=row.cv_f1, bold=idx in cv_f1_max),
            fmt_value(value=row.cv_f1_std, bold=False),
            fmt_value(value=row.test_f1, bold=idx in test_f1_max),
            fmt_value(value=row.test_prec, bold=idx in test_prec_max),
            fmt_value(value=row.test_rec, bold=idx in test_rec_max),
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


def fmt_number(*, value: float, decimals: int = METRIC_DECIMALS, bold: bool = False) -> str:
    """Format a percentage value with fixed decimals."""
    if pd.isna(value):
        return '--'
    if abs(value) < (0.5 * (10 ** -decimals)):
        value = 0.0
    text = format(value, f'.{decimals}f')
    return f'\\textbf{{{text}}}' if bold else text


def fmt_plain(*, value: float, decimals: int = METRIC_DECIMALS) -> str:
    """Format a unit-interval metric as a percentage."""
    return fmt_number(value=to_pct(value=value), decimals=decimals)


def fmt_delta(*, value: float, bold: bool, decimals: int = METRIC_DECIMALS) -> str:
    """Format a unit-interval metric delta as a percentage point change."""
    return fmt_number(value=to_pct(value=value), decimals=decimals, bold=bold)


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
        '& \\multicolumn{2}{c}{\\textbf{Precision (\\%)}}',
        '& \\multicolumn{2}{c}{\\textbf{Recall (\\%)}}',
        '& \\multicolumn{2}{c}{\\textbf{F1 Score (\\%)}} \\\\',
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
    ('support', 'Support 0--15', 'macro_support_0-15'),
    ('support', 'Support 15--100', 'macro_support_15-100'),
    ('support', 'Support 100--1000', 'macro_support_100-1000'),
    ('support', 'Support 1000+', 'macro_support_1000+'),
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
        '& \\multicolumn{2}{c}{\\textbf{Precision (\\%)}}',
        '& \\multicolumn{2}{c}{\\textbf{Recall (\\%)}}',
        '& \\multicolumn{2}{c}{\\textbf{F1 Score (\\%)}} \\\\',
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



def clean_category_name(*, raw: str) -> str:
    """Strip surrounding quotes, normalise whitespace, and wrap hierarchy arrows in math mode."""
    text = str(raw).strip().strip('"')
    text = re.sub(r'>{2,}', lambda m: f'${m.group()}$', text)
    return text


def render_mcnemar_significant_table(
    *,
    experiment: ExperimentEntry,
    df: pd.DataFrame,
    direction: str,
) -> str:
    """Render a table listing all categories that passed the McNemar test.

    :param experiment: experiment metadata
    :param df: dataframe already filtered to ``mcnemar_pass == 1``
    :param direction: ``'improved'`` or ``'degraded'``
    :return: LaTeX table string
    """
    ascending = direction == 'degraded'
    df = df.sort_values('F1_diff', ascending=ascending).reset_index(drop=True)

    lines: list[str] = [
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrrrrr}',
        '\\toprule',
        '\\textbf{IPTC Category}',
        '& \\textbf{Support}',
        '& \\textbf{F1 Base (\\%)}',
        '& \\textbf{F1 Current (\\%)}',
        '& \\textbf{$\\Delta$F1 (\\%)}',
        '& \\textbf{p-value (FDR)} \\\\',
        '\\midrule',
    ]

    for _, row in df.iterrows():
        category = escape_tag(tag=clean_category_name(raw=row['IPTC Category']))
        support = int(as_float(value=row['article_frequency']))
        f1_base = fmt_plain(value=as_float(value=row['F1_base']))
        f1_cur = fmt_plain(value=as_float(value=row['F1_current']))
        f1_diff = fmt_delta(value=as_float(value=row['F1_diff']), bold=True)
        p_val = as_float(value=row['mcnemar_p_value_fdr'])
        p_str = f'{p_val:.4f}' if pd.notna(p_val) else '--'
        lines.append(f'{category} & {support} & {f1_base} & {f1_cur} & {f1_diff} & {p_str} \\\\')

    dir_label = 'improved' if direction == 'improved' else 'degraded'
    caption = (
        f'All statistically significant {dir_label} categories for '
        f'\\textit{{{escape_tag(tag=experiment.display_tag)}}} vs.\\ the base model '
        f'(McNemar test, FDR-corrected $p < 0.05$). '
        f'Categories are sorted by $\\Delta$F1 in {"descending" if direction == "improved" else "ascending"} order.'
    )
    label = f'tab:{sanitize_token(text=experiment.display_tag)}-mcnemar-{dir_label}'
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


def write_per_experiment_tables(*, experiments: Sequence[ExperimentEntry], output_dir: Path) -> int:
    """Generate and write per-experiment LaTeX tables.

    Produces corpora, group summary, and McNemar-significant improved/degraded tables.
    """
    written = 0
    for experiment in experiments:
        corpora_path = experiment.subdir / CORPORA_FILENAME
        summary_path = experiment.subdir / SUMMARY_FILENAME
        improved_path = experiment.subdir / TOP_IMPROVED_FILENAME
        degraded_path = experiment.subdir / TOP_DEGRADED_FILENAME
        required_paths = (corpora_path, summary_path, improved_path, degraded_path)
        if not all(path.is_file() for path in required_paths):
            LOG.warning(
                f'Skipping per-experiment tables for {experiment.subdir.name}: '
                'required CSV files are missing.'
            )
            continue

        corpora_df = pd.read_csv(corpora_path)
        summary_df = pd.read_csv(summary_path)
        improved_df = pd.read_csv(improved_path)
        degraded_df = pd.read_csv(degraded_path)

        improved_sig = improved_df[improved_df['mcnemar_pass'] == 1]
        degraded_sig = degraded_df[degraded_df['mcnemar_pass'] == 1]

        file_stem = sanitize_token(text=experiment.subdir.name)

        corpora_tex = render_experiment_corpora_table(experiment=experiment, corpora_df=corpora_df)
        summary_tex = render_experiment_group_summary_table(experiment=experiment, summary_df=summary_df)
        (output_dir / f'{file_stem}_corpora.tex').write_text(corpora_tex, encoding='utf-8')
        (output_dir / f'{file_stem}_summary_groups.tex').write_text(summary_tex, encoding='utf-8')
        written += 2

        if not improved_sig.empty:
            improved_tex = render_mcnemar_significant_table(
                experiment=experiment, df=improved_sig, direction='improved',
            )
            (output_dir / f'{file_stem}_mcnemar_improved.tex').write_text(improved_tex, encoding='utf-8')
            written += 1
            LOG.info(f'{experiment.display_tag}: {len(improved_sig)} McNemar-significant improved categories')
        else:
            LOG.info(f'{experiment.display_tag}: no McNemar-significant improved categories')

        if not degraded_sig.empty:
            degraded_tex = render_mcnemar_significant_table(
                experiment=experiment, df=degraded_sig, direction='degraded',
            )
            (output_dir / f'{file_stem}_mcnemar_degraded.tex').write_text(degraded_tex, encoding='utf-8')
            written += 1
            LOG.info(f'{experiment.display_tag}: {len(degraded_sig)} McNemar-significant degraded categories')
        else:
            LOG.info(f'{experiment.display_tag}: no McNemar-significant degraded categories')

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
