'''
Generate publication-quality thesis graphs from a comparison Excel workbook.

Reads the evaluation comparison Excel produced by
:mod:`iptc_entity_pipeline.evaluation.comparison` and generates bar charts,
moving-average line plots, and IPTC topic-level comparisons suitable for
direct inclusion in a LaTeX thesis document.
'''

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display-name mappings
# ---------------------------------------------------------------------------

CORPORA_NAMES: dict[str, str] = {
    'cs_mafra_iptc': 'Mafra (CS)',
    'de_apa': 'APA (DE)',
    'de_dpa_iptc': 'DPA (DE)',
    'de_eurosport_iptc': 'Es (DE)',
    'en_bbc_iptc': 'BBC (EN)',
    'en_cnn_iptc': 'CNN (EN)',
    'en_dw_iptc': 'DW (EN)',
    'en_eurosport_iptc': 'Es (EN)',
    'en_mediahuis_iptc': 'Mediahuis (EN)',
    'es_eurosport_iptc': 'Es (ES)',
    'fr_rts_instagram_iptc': 'RTS Ing (FR)',
    'fr_rts_iptc': 'RTS (FR)',
    'nl_eurosport_iptc': 'Es (NL)',
    'nl_eventdna': 'DNA (NL)',
    'nl_mediahuis_iptc': 'Mediahuis (NL)',
    'nl_noordhollandsdagblad': 'NoordHB (NL)',
    'nl_nu_iptc': 'NU (NL)',
}

LANG_NAMES: dict[str, str] = {
    'cs': 'Czech',
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'nl': 'Dutch',
}

SUMMARY_ROWS: set[str] = {'All_datapoint', 'All_macro_corpora', 'All_micro'}

_METRIC_COLORS: dict[str, str] = {
    'Precision_diff': '#4178BC',
    'Recall_diff': '#59A14F',
    'F1_diff': '#E8871E',
}

_METRIC_LABELS: dict[str, str] = {
    'Precision_diff': 'Precision',
    'Recall_diff': 'Recall',
    'F1_diff': 'F1-score',
}


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def _configure_mpl() -> None:
    '''Set matplotlib rcParams for thesis-quality serif figures.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F401

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def _set_log_number_ticks(axis: object) -> None:
    '''Show plain integer numbers (not powers of ten) on a logarithmic axis.'''
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    axis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))  # type: ignore[union-attr]
    axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))  # type: ignore[union-attr]
    axis.set_major_formatter(fmt)  # type: ignore[union-attr]
    axis.set_minor_formatter(NullFormatter())  # type: ignore[union-attr]


def _save_fig(fig: object, output_dir: Path, name: str) -> tuple[Path, Path]:
    '''Save figure as both PDF and PNG, return (pdf_path, png_path).'''
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f'{name}.pdf'
    png = output_dir / f'{name}.png'
    fig.savefig(pdf)  # type: ignore[union-attr]
    fig.savefig(png)  # type: ignore[union-attr]
    LOG.info(f'Saved {pdf} and {png}')
    return pdf, png


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sheet(*, excel_path: Path, sheet: str) -> pd.DataFrame:
    '''Load a single sheet from the comparison Excel workbook.

    :param excel_path: path to the .xlsx file
    :param sheet: worksheet name
    :return: DataFrame with the sheet contents
    '''
    LOG.info(f'Loading sheet={sheet} from {excel_path}')
    return pd.read_excel(excel_path, sheet_name=sheet)


def load_corpora(*, excel_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    '''Load corpora comparison sheet and extract the All_micro reference row.

    :return: (corpora_df excluding summary rows, all_micro series)
    '''
    df = load_sheet(excel_path=excel_path, sheet='corpora_comparison')
    micro = df[df['Corpus Name'] == 'All_micro'].iloc[0]
    corpora = df[~df['Corpus Name'].isin(SUMMARY_ROWS)].copy()
    return corpora, micro


def load_languages(*, excel_path: Path) -> pd.DataFrame:
    '''Load language comparison sheet.'''
    return load_sheet(excel_path=excel_path, sheet='language_comparison')


def load_classes(*, excel_path: Path) -> pd.DataFrame:
    '''Load classes comparison sheet.'''
    return load_sheet(excel_path=excel_path, sheet='classes_comparison')


def load_thresholds(*, excel_path: Path) -> pd.DataFrame:
    '''Load class thresholds sheet.'''
    return load_sheet(excel_path=excel_path, sheet='class_thresholds')


def _filter_subclasses(df: pd.DataFrame, *, min_support: int) -> pd.DataFrame:
    '''Keep only subclasses (exclude top-level parents) with sufficient support, sorted by support ascending.

    Top-level classes use ``" - "`` in their display name (e.g. ``arts+ - arts, ...``).
    Aggregate rows (``All_*``) are also excluded.
    '''
    is_toplevel = df['IPTC Category'].str.contains(' - ', na=False)
    is_aggregate = df['IPTC Category'].str.startswith('All_', na=False)
    filtered = df[~is_toplevel & ~is_aggregate & (df['Data Count_current'] >= min_support)].copy()
    return filtered.sort_values('Data Count_current').reset_index(drop=True)


# ---------------------------------------------------------------------------
# top_level_from_label (reused from comparison.py)
# ---------------------------------------------------------------------------

def top_level_from_label(label: str) -> str:
    '''Extract IPTC top-level category prefix from a quoted long-name label.

    :param label: e.g. ``'"arts+ > culture (20000038)"'``
    :return: top-level prefix, e.g. ``'arts+'``
    '''
    inner = str(label).strip().strip('"').strip()
    delim_positions = [pos for pos in (inner.find(' >'), inner.find(' -')) if pos != -1]
    if delim_positions:
        return inner[: min(delim_positions)].strip()
    paren_idx = inner.find('(')
    return inner[:paren_idx].strip() if paren_idx != -1 else inner


# ---------------------------------------------------------------------------
# Graph 1a / 1b: Corpora bar charts
# ---------------------------------------------------------------------------

def _plot_grouped_bars(
    *,
    df: pd.DataFrame,
    name_col: str,
    name_map: dict[str, str],
    micro: pd.Series,
    title: str,
    fig_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    '''Grouped triple-bar chart with horizontal reference lines.

    :return: (pdf_path, png_path)
    '''
    import matplotlib.pyplot as plt

    metrics = ['Precision_diff', 'Recall_diff', 'F1_diff']
    labels = [name_map.get(str(r), str(r)) for r in df[name_col]]
    x = np.arange(len(labels))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, metric in enumerate(metrics):
        vals = df[metric].values * 100
        ax.bar(
            x + (i - 1) * bar_w,
            vals,
            width=bar_w,
            label=_METRIC_LABELS[metric],
            color=_METRIC_COLORS[metric],
            edgecolor='white',
            linewidth=0.4,
        )

    ref_val = float(micro['F1_diff']) * 100
    ax.axhline(
        ref_val,
        color=_METRIC_COLORS['F1_diff'],
        linewidth=0.8,
        linestyle='--',
        alpha=0.7,
        label=f'Micro avg F1 ({ref_val:.1f}%)',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Difference (%)')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    paths = _save_fig(fig, output_dir, fig_name)
    plt.close(fig)
    return paths


def plot_corpora(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 1a and 1b (corpora bar charts).

    :return: list of (figure_name, pdf_path, png_path)
    '''
    corpora, micro = load_corpora(excel_path=excel_path)
    corpora = corpora.sort_values('F1_diff', ascending=False).reset_index(drop=True)

    top9 = corpora.iloc[:9]
    bot8 = corpora.iloc[9:]

    results = []
    pdf, png = _plot_grouped_bars(
        df=top9,
        name_col='Corpus Name',
        name_map=CORPORA_NAMES,
        micro=micro,
        title='Per-corpus performance difference (top 9)',
        fig_name='graph1a_corpora_top9',
        output_dir=output_dir,
    )
    results.append(('graph1a_corpora_top9', pdf, png))

    pdf, png = _plot_grouped_bars(
        df=bot8,
        name_col='Corpus Name',
        name_map=CORPORA_NAMES,
        micro=micro,
        title='Per-corpus performance difference (remaining 8)',
        fig_name='graph1b_corpora_bottom8',
        output_dir=output_dir,
    )
    results.append(('graph1b_corpora_bottom8', pdf, png))
    return results


# ---------------------------------------------------------------------------
# Graph 2: Language bar chart
# ---------------------------------------------------------------------------

def plot_languages(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 2 (language bar chart).'''
    langs = load_languages(excel_path=excel_path)
    langs = langs.sort_values('F1_diff', ascending=False).reset_index(drop=True)

    _, micro = load_corpora(excel_path=excel_path)

    pdf, png = _plot_grouped_bars(
        df=langs,
        name_col='Language',
        name_map=LANG_NAMES,
        micro=micro,
        title='Per-language performance difference',
        fig_name='graph2_languages',
        output_dir=output_dir,
    )
    return [('graph2_languages', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 3: F1 difference scatter with linear regression
# ---------------------------------------------------------------------------

def plot_f1_diff_scatter(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 3 (scatter of F1 diff by class rank with linear regression).'''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    df = _filter_subclasses(classes, min_support=15)

    x = np.arange(len(df))
    y = df['F1_diff'].values * 100

    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y, s=8, alpha=0.4, color='#4178BC', edgecolors='none')
    ax.plot(x, trend(x), color='#E15759', linewidth=1.5, label=f'Linear fit (slope={coeffs[0]:.3f})')

    ax.set_ylim(-10, 10)
    ax.set_xlabel('Class rank (by support)')
    ax.set_ylabel('F1-score difference (%)')
    ax.set_title('Per-class F1-score difference vs. class support rank')
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph3_f1_diff_scatter')
    plt.close(fig)
    return [('graph3_f1_diff_scatter', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 4: Absolute F1 scatter with linear regression (two classifiers)
# ---------------------------------------------------------------------------

def plot_f1_abs_scatter(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 4 (scatter of absolute F1 for both models with regression).'''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    df = _filter_subclasses(classes, min_support=15)

    x = np.arange(len(df))
    y_current = df['F1_current'].values * 100
    y_base = df['F1_base'].values * 100

    coeffs_cur = np.polyfit(x, y_current, 1)
    coeffs_base = np.polyfit(x, y_base, 1)
    trend_cur = np.poly1d(coeffs_cur)
    trend_base = np.poly1d(coeffs_base)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y_current, s=8, alpha=0.35, color='#4178BC', edgecolors='none', label='Current model')
    ax.scatter(x, y_base, s=8, alpha=0.35, color='#E8871E', edgecolors='none', label='Base model')
    ax.plot(x, trend_cur(x), color='#4178BC', linewidth=1.5, linestyle='-')
    ax.plot(x, trend_base(x), color='#E8871E', linewidth=1.5, linestyle='-')

    ax.set_ylim(20, 90)
    ax.set_xlabel('Class rank (by support)')
    ax.set_ylabel('F1-score (%)')
    ax.set_title('Per-class F1-score vs. class support rank')
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph4_f1_abs_scatter')
    plt.close(fig)
    return [('graph4_f1_abs_scatter', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 3b: F1 difference moving average
# ---------------------------------------------------------------------------

def _find_support_boundary(df: pd.DataFrame, *, threshold: int, support_col: str = 'Data Count_current') -> int | None:
    '''Find index where support first reaches *threshold*.'''
    above = df[df[support_col] >= threshold]
    if above.empty:
        return None
    return int(above.index[0])


def plot_f1_diff_moving_avg(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 3b (moving average F1 diff by class rank).'''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    df = _filter_subclasses(classes, min_support=15)

    rolling = (df['F1_diff'] * 100).rolling(window=70, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(rolling)), rolling.values, color='#4178BC', linewidth=1.2)

    boundary = _find_support_boundary(df, threshold=75)
    if boundary is not None:
        ax.axvline(boundary, color='red', linewidth=0.8, linestyle='-', alpha=0.8)

    ax.set_xlabel('Class rank (by support)')
    ax.set_ylabel('F1-score difference (%)')
    ax.set_title('Moving average of F1-score difference (window = 70)')
    ax.set_xticks([])
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph3b_f1_diff_moving_avg')
    plt.close(fig)
    return [('graph3b_f1_diff_moving_avg', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 4b: Absolute F1 moving average (two classifiers)
# ---------------------------------------------------------------------------

def plot_f1_abs_moving_avg(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 4b (moving average absolute F1 for both models).'''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    df = _filter_subclasses(classes, min_support=15)

    support = df['Data Count_current'].values
    roll_current = (df['F1_current'] * 100).rolling(window=70, min_periods=10).mean()
    roll_base = (df['F1_base'] * 100).rolling(window=70, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(support, roll_current.values, color='#4178BC', linewidth=1.2, label='Current model')
    ax.plot(support, roll_base.values, color='#E8871E', linewidth=1.2, label='Base model')

    ax.set_xscale('log')
    _set_log_number_ticks(ax.xaxis)
    ax.set_xlabel('Class support (log scale)')
    ax.set_ylabel('F1-score (%)')
    ax.set_title('Moving average of F1-score (window = 70)')
    ax.legend(fontsize=8)
    ax.grid(axis='both', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph4b_f1_abs_moving_avg')
    plt.close(fig)
    return [('graph4b_f1_abs_moving_avg', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 8: F1 difference moving average vs. support (log scale)
# ---------------------------------------------------------------------------

def plot_f1_diff_vs_support(
    *,
    excel_path: Path,
    output_dir: Path,
    window: int = 50,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 8 (moving average of absolute F1 diff vs. support, log x-axis).

    Classes are sorted by support ascending, a rolling mean of the absolute F1-score
    difference is computed, and it is plotted against the actual support value on a
    logarithmic x-axis to reveal how the average magnitude of change evolves with
    class frequency.
    '''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    df = _filter_subclasses(classes, min_support=15)

    support = df['Data Count_current'].values
    rolling = (df['F1_diff'].abs() * 100).rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(support, rolling.values, color='#4178BC', linewidth=1.2)

    ax.set_xscale('log')
    ax.set_xlabel('Class support (log scale)')
    ax.set_ylabel('Absolute F1-score difference (%)')
    ax.set_title(f'Moving average of absolute F1-score difference vs. class support (window = {window})')
    ax.grid(axis='both', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph8_f1_diff_vs_support')
    plt.close(fig)
    return [('graph8_f1_diff_vs_support', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 5a/5b: Threshold scatter with linear regression
# ---------------------------------------------------------------------------

def _plot_threshold_rank(
    *,
    rank: np.ndarray,
    threshold: np.ndarray,
    output_dir: Path,
) -> tuple[str, Path, Path]:
    '''Scatter of per-class threshold vs. support rank (linear axes).'''
    import matplotlib.pyplot as plt

    trend = np.poly1d(np.polyfit(rank, threshold, 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(rank, threshold, s=8, alpha=0.4, color='#4178BC', edgecolors='none')
    ax.plot(rank, trend(rank), color='#E15759', linewidth=1.5, label='Linear fit')
    ax.axhline(0.25, color='red', linewidth=0.8, linestyle='--', alpha=0.8, label='Min threshold (0.25)')

    ax.set_ylabel('Threshold')
    ax.set_title('Per-class classification threshold vs. support rank')
    ax.set_xlabel('Class rank (by support)')
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph5_linear_threshold_scatter')
    plt.close(fig)
    return 'graph5_linear_threshold_scatter', pdf, png


def _plot_threshold_support(
    *,
    support: np.ndarray,
    threshold: np.ndarray,
    output_dir: Path,
    window: int = 20,
) -> tuple[str, Path, Path]:
    '''Scatter of per-class threshold vs. support on a logarithmic x-axis.

    A rolling mean of the threshold (classes sorted by support ascending) is overlaid
    on the scatter to reveal the trend with class frequency.
    '''
    import matplotlib.pyplot as plt

    rolling = pd.Series(threshold).rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(support, threshold, s=8, alpha=0.4, color='#4178BC', edgecolors='none')
    ax.plot(support, rolling.values, color='#E15759', linewidth=1.5, label=f'Rolling mean (window = {window})')
    ax.axhline(0.25, color='red', linewidth=0.8, linestyle='--', alpha=0.8, label='Min threshold (0.25)')

    ax.set_xscale('log')
    _set_log_number_ticks(ax.xaxis)
    ax.set_ylabel('Threshold')
    ax.set_xlabel('Class support (log scale)')
    ax.set_title('Per-class classification threshold vs. class support')
    ax.legend(fontsize=8)
    ax.grid(axis='both', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph5_log_threshold_scatter')
    plt.close(fig)
    return 'graph5_log_threshold_scatter', pdf, png


def plot_threshold_scatter(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 5a (rank, linear) and 5b (support, log) threshold scatter plots.'''
    thr = load_thresholds(excel_path=excel_path)
    thr = thr[thr['count'] >= 15].sort_values('count').reset_index(drop=True)

    rank = np.arange(len(thr))
    support = thr['count'].values.astype(float)
    threshold = thr['threshold_current'].values

    results = [
        _plot_threshold_rank(rank=rank, threshold=threshold, output_dir=output_dir),
        _plot_threshold_support(support=support, threshold=threshold, output_dir=output_dir),
    ]
    return results


# ---------------------------------------------------------------------------
# Graph 6: IPTC top-level category F1 improvement
# ---------------------------------------------------------------------------

def _topic_full_names(classes: pd.DataFrame) -> dict[str, str]:
    '''Map each IPTC top-level prefix to its full descriptive name.

    Top-level rows look like ``"arts+ - arts, culture, entertainment and media (01000000)"``;
    the prefix is ``arts+`` and the full name is the text between ``" - "`` and the id.
    '''
    names: dict[str, str] = {}
    toplevel = classes[classes['IPTC Category'].str.contains(' - ', na=False)]
    for label in toplevel['IPTC Category']:
        inner = str(label).strip().strip('"').strip()
        prefix, _, rest = inner.partition(' - ')
        full = rest.rsplit('(', 1)[0].strip()
        names[prefix.strip()] = full or prefix.strip()
    return names


def plot_iptc_topic_comparison(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 6 (IPTC topic-level macro F1 improvement).'''
    import matplotlib.pyplot as plt

    classes = load_classes(excel_path=excel_path)
    full_names = _topic_full_names(classes)
    subs = _filter_subclasses(classes, min_support=15)

    subs = subs.copy()
    subs['topic'] = subs['IPTC Category'].apply(top_level_from_label)

    grouped = subs.groupby('topic').agg(
        avg_f1_diff=('F1_diff', 'mean'),
        n_classes=('F1_diff', 'count'),
    ).reset_index()
    grouped['avg_f1_diff'] *= 100
    grouped = grouped.sort_values('avg_f1_diff', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#59A14F' if v >= 0 else '#E15759' for v in grouped['avg_f1_diff']]
    bars = ax.bar(range(len(grouped)), grouped['avg_f1_diff'], color=colors, edgecolor='white', linewidth=0.4)

    for bar, n in zip(bars, grouped['n_classes']):
        y = bar.get_height()
        offset = 0.15 if y >= 0 else -0.15
        va = 'bottom' if y >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset, str(int(n)),
                ha='center', va=va, fontsize=10, fontweight='bold')

    top = float(grouped['avg_f1_diff'].max())
    bottom = float(grouped['avg_f1_diff'].min())
    ax.set_ylim(bottom - 0.8, top + 0.8)

    labels = [full_names.get(t, t) for t in grouped['topic']]
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=12)
    ax.set_ylabel('F1-score difference (%)')
    ax.set_title('Average F1-score improvement by IPTC topic')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    pdf, png = _save_fig(fig, output_dir, 'graph6_iptc_topic_comparison')
    plt.close(fig)
    return [('graph6_iptc_topic_comparison', pdf, png)]


# ---------------------------------------------------------------------------
# Graph 7a/7b: Entity type proportion pie charts
# ---------------------------------------------------------------------------

_ENTITY_TYPE_COLORS: dict[str, str] = {
    'person': '#4178BC',
    'location': '#59A14F',
    'organization': '#E8871E',
    'general': '#A0A0A0',
    'product': '#9B59B6',
    'event': '#E15759',
}
_ENTITY_TYPE_ORDER: tuple[str, ...] = (
    'person',
    'location',
    'organization',
    'general',
    'product',
    'event',
)

_MINOR_TYPES: set[str] = {'idiom', 'simile', 'namedphrase', 'other'}


def _build_pie_data(
    df: pd.DataFrame,
    *,
    weight_col: str | None,
) -> tuple[list[str], list[float]]:
    '''Aggregate entity types in a fixed semantic order.

    :param df: filtered entity DataFrame
    :param weight_col: column for weighted aggregation, or None for count
    :return: (labels, values) sorted descending
    '''
    df = df[~df['entity_type'].isin(_MINOR_TYPES)].copy()

    if weight_col is not None:
        grouped = df.groupby('entity_type')[weight_col].sum().abs()
    else:
        grouped = df.groupby('entity_type').size()

    grouped = grouped.reindex(_ENTITY_TYPE_ORDER).dropna()
    grouped = grouped[grouped > 0]
    return list(grouped.index), list(grouped.values)


def _plot_entity_pie_triple(
    *,
    pos: pd.DataFrame,
    zero: pd.DataFrame,
    neg: pd.DataFrame,
    weight_col: str | None,
    title: str,
    fig_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    '''Three pie charts: positive, zero, and negative entity impact.'''
    import matplotlib.pyplot as plt

    fig, (ax_pos, ax_zero, ax_neg) = plt.subplots(1, 3, figsize=(14, 4.5))

    groups = [
        (ax_pos, pos, 'Positive impact', weight_col),
        (ax_zero, zero, 'Zero impact', None),
        (ax_neg, neg, 'Negative impact', weight_col),
    ]
    for ax, data, subtitle, wcol in groups:
        labels, values = _build_pie_data(data, weight_col=wcol)
        colors = [_ENTITY_TYPE_COLORS.get(l, '#CCCCCC') for l in labels]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            pctdistance=0.75,
            startangle=90,
            textprops={'fontsize': 10},
        )
        for t in autotexts:
            t.set_fontsize(7)
        title_size = 12 
        ax.set_title(subtitle, fontsize=title_size)

    fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()

    paths = _save_fig(fig, output_dir, fig_name)
    plt.close(fig)
    return paths


def plot_entity_type_pies(
    *,
    excel_path: Path,
    output_dir: Path,
) -> list[tuple[str, Path, Path]]:
    '''Generate Graph 7a (count) and 7b (score-weighted) entity type pies.'''
    entities = load_sheet(excel_path=excel_path, sheet='entity_impact_all')
    pos = entities[entities['entity_score'] > 0]
    zero = entities[entities['entity_score'] == 0]
    neg = entities[entities['entity_score'] < 0]

    results = []

    pdf, png = _plot_entity_pie_triple(
        pos=pos, zero=zero, neg=neg,
        weight_col=None,
        title='Entity type distribution by impact direction (count)',
        fig_name='graph7a_entity_type_count',
        output_dir=output_dir,
        
    )
    results.append(('graph7a_entity_type_count', pdf, png))

    pdf, png = _plot_entity_pie_triple(
        pos=pos, zero=zero, neg=neg,
        weight_col='entity_score',
        title='Entity type distribution by impact direction (weighted by score)',
        fig_name='graph7b_entity_type_weighted',
        output_dir=output_dir,
    )
    results.append(('graph7b_entity_type_weighted', pdf, png))

    return results


# ---------------------------------------------------------------------------
# LaTeX figure generation
# ---------------------------------------------------------------------------

_LATEX_META: dict[str, tuple[str, str]] = {
    'graph1a_corpora_top9': (
        'Per-corpus performance difference between the entity-enhanced and base classifier '
        '(top~9 corpora by F1-score improvement). Dashed horizontal lines indicate the '
        'micro-averaged difference across all corpora.',
        'corpora-top9',
    ),
    'graph1b_corpora_bottom8': (
        'Per-corpus performance difference between the entity-enhanced and base classifier '
        '(remaining 8 corpora). Dashed horizontal lines indicate the micro-averaged difference '
        'across all corpora.',
        'corpora-bottom8',
    ),
    'graph2_languages': (
        'Per-language macro-averaged performance difference between the entity-enhanced and '
        'base classifier. Dashed horizontal lines indicate the micro-averaged difference.',
        'languages',
    ),
    'graph3_f1_diff_scatter': (
        'Per-class F1-score difference between the entity-enhanced and base classifier '
        'plotted against class support rank (ascending). The regression line shows the '
        'overall trend of diminishing improvement for higher-support classes.',
        'f1-diff-scatter',
    ),
    'graph4_f1_abs_scatter': (
        'Per-class absolute F1-score for both classifiers plotted against class support '
        'rank (ascending), with separate linear regression fits. The gap between the '
        'regression lines reflects the average improvement across the support spectrum.',
        'f1-abs-scatter',
    ),
    'graph3b_f1_diff_moving_avg': (
        'Moving average (window~=~70) of per-class F1-score difference sorted by class '
        'support. The red vertical line marks the boundary at support~=~75.',
        'f1-diff-moving-avg',
    ),
    'graph4b_f1_abs_moving_avg': (
        'Moving average (window~=~70) of absolute F1-score for both classifiers plotted '
        'against class support on a logarithmic x-axis.',
        'f1-abs-moving-avg',
    ),
    'graph5_linear_threshold_scatter': (
        'Per-class classification threshold plotted against class support rank with a '
        'linear regression fit. The red dashed line marks the minimum threshold of~0.25.',
        'threshold-linear',
    ),
    'graph5_log_threshold_scatter': (
        'Per-class classification threshold plotted against class support on a logarithmic '
        'x-axis, with a rolling mean (window~=~20, classes sorted by ascending support) '
        'overlaid to show the trend. The red dashed line marks the minimum threshold of~0.25.',
        'threshold-log',
    ),
    'graph6_iptc_topic_comparison': (
        'Macro-averaged F1-score improvement per IPTC top-level topic. Numbers above bars '
        'indicate the count of qualifying subclasses (support~$\\geq$~15).',
        'iptc-topics',
    ),
    'graph7a_entity_type_count': (
        'Distribution of entity types among entities with positive (left), zero (centre), '
        'and negative (right) impact on classification, where each entity counts equally.',
        'entity-type-count',
    ),
    'graph7b_entity_type_weighted': (
        'Distribution of entity types among entities with positive (left), zero (centre), '
        'and negative (right) impact on classification. Positive and negative groups are '
        'weighted by absolute entity impact score; the zero group uses entity counts.',
        'entity-type-weighted',
    ),
    'graph8_f1_diff_vs_support': (
        'Moving average (window~=~50) of the per-class absolute F1-score difference between '
        'the entity-enhanced and base classifier, plotted against class support on a '
        'logarithmic axis. Shows how the average magnitude of change evolves with class '
        'frequency.',
        'f1-diff-vs-support',
    ),
}


def write_latex(*, figures: Sequence[tuple[str, Path, Path]], output_dir: Path) -> Path:
    '''Write LaTeX figure snippets for every generated graph.

    :return: path to the latex output directory
    '''
    latex_dir = output_dir / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    for fig_name, pdf_path, _ in figures:
        caption, label = _LATEX_META.get(fig_name, (fig_name, fig_name))
        tex = textwrap.dedent(f'''\
            \\begin{{figure}}[htbp]
              \\centering
              \\includegraphics[width=\\textwidth]{{{pdf_path.name}}}
              \\caption{{{caption}}}
              \\label{{fig:{label}}}
            \\end{{figure}}
        ''')
        tex_path = latex_dir / f'{fig_name}.tex'
        tex_path.write_text(tex, encoding='utf-8')
        LOG.info(f'Wrote LaTeX snippet {tex_path}')

    return latex_dir


# ---------------------------------------------------------------------------
# Markdown documentation
# ---------------------------------------------------------------------------

_MD_SECTIONS: dict[str, str] = {
    'graph1a_corpora_top9': textwrap.dedent('''\
        ### Graph 1a -- Per-corpus performance difference (top 9)

        **Source:** `corpora_comparison` sheet, columns `Precision_diff`, `Recall_diff`, `F1_diff`.

        **Methodology:** The 17 individual corpora (summary rows excluded) are sorted by
        F1-score difference in descending order. The top 9 are displayed as grouped bar charts
        with three bars per corpus (Precision, Recall, F1). Thin dashed horizontal lines show
        the micro-averaged difference from the `All_micro` row.

        **Interpretation:** Positive bars indicate improvement of the entity-enhanced model
        over the baseline. The dashed lines provide a global reference point.
    '''),
    'graph1b_corpora_bottom8': textwrap.dedent('''\
        ### Graph 1b -- Per-corpus performance difference (remaining 8)

        **Source:** Same as Graph 1a.

        **Methodology:** The remaining 8 corpora (lowest F1 improvement or degradation).

        **Interpretation:** Corpora with near-zero or negative bars show where the
        entity-enhanced model performs similarly or worse than the baseline.
    '''),
    'graph2_languages': textwrap.dedent('''\
        ### Graph 2 -- Per-language performance difference

        **Source:** `language_comparison` sheet, columns `Precision_diff`, `Recall_diff`, `F1_diff`.

        **Methodology:** Macro average over corpora grouped by language prefix. Sorted by
        F1-score difference. Dashed horizontal lines from the `All_micro` row of
        `corpora_comparison`.

        **Interpretation:** Shows which languages benefit most from entity enhancement.
    '''),
    'graph3_f1_diff_scatter': textwrap.dedent('''\
        ### Graph 3 -- Per-class F1-score difference scatter

        **Source:** `classes_comparison` sheet, column `F1_diff`.

        **Methodology:** Subclasses with support >= 15 are sorted by support ascending and
        plotted as a scatter chart. A linear regression line is fitted to show the overall
        trend across the support spectrum.

        **Interpretation:** Each dot is one IPTC class. The regression slope indicates
        whether the improvement is larger for rare or frequent classes.
    '''),
    'graph4_f1_abs_scatter': textwrap.dedent('''\
        ### Graph 4 -- Per-class absolute F1-score scatter

        **Source:** `classes_comparison` sheet, columns `F1_current`, `F1_base`.

        **Methodology:** Same filtering and sorting as Graph 3. Two sets of scatter points
        are plotted (current and base model) with separate linear regression fits.

        **Interpretation:** The vertical gap between the two regression lines shows the
        average improvement. The slopes show how both models scale with class frequency.
    '''),
    'graph3b_f1_diff_moving_avg': textwrap.dedent('''\
        ### Graph 3b -- Moving average of F1-score difference

        **Source:** `classes_comparison` sheet, column `F1_diff`.

        **Methodology:** Subclasses with support >= 15 are sorted by support ascending.
        A rolling mean with window 70 (min_periods=10) is computed over F1_diff.
        A red vertical line marks the transition at support = 75.

        **Interpretation:** Shows how the F1-score improvement evolves from rare to frequent
        classes. The moving average smooths out individual class noise.
    '''),
    'graph4b_f1_abs_moving_avg': textwrap.dedent('''\
        ### Graph 4b -- Moving average of absolute F1-score

        **Source:** `classes_comparison` sheet, columns `F1_current`, `F1_base`.

        **Methodology:** Same filtering and sorting as Graph 3b. Two rolling means (window 70)
        are plotted: one for the current (entity-enhanced) model and one for the baseline.
        The x-axis is the actual class support on a logarithmic scale.

        **Interpretation:** Visualizes the gap between the two models across the support
        spectrum and shows overall performance levels against class frequency.
    '''),
    'graph5_linear_threshold_scatter': textwrap.dedent('''\
        ### Graph 5a -- Per-class classification threshold scatter (linear scale)

        **Source:** `class_thresholds` sheet, column `threshold_current`.

        **Methodology:** Classes with support >= 15 sorted by support ascending. Each dot
        is one class's optimised threshold. A linear regression line is fitted. The red
        dashed horizontal line marks the minimum threshold of 0.25.

        **Interpretation:** Shows how the optimised per-class threshold varies with class
        frequency. The regression slope indicates whether higher-support classes tend to
        have higher thresholds.
    '''),
    'graph5_log_threshold_scatter': textwrap.dedent('''\
        ### Graph 5b -- Per-class classification threshold vs. support (log x-axis)

        **Source:** Same as Graph 5a.

        **Methodology:** The optimised per-class threshold is plotted against the actual class
        support on a logarithmic x-axis (real support numbers are shown as tick labels).
        Classes are sorted by ascending support and a rolling mean (window 20) of the
        threshold is overlaid on the scatter.

        **Interpretation:** Reveals how the optimised threshold scales with class frequency,
        with the rolling mean smoothing out individual class noise.
    '''),
    'graph6_iptc_topic_comparison': textwrap.dedent('''\
        ### Graph 6 -- IPTC top-level topic F1 improvement

        **Source:** `classes_comparison` sheet, column `F1_diff`.

        **Methodology:** Subclasses with support >= 15 are grouped by their IPTC top-level
        topic prefix (e.g. "arts+", "sport"). The macro average of F1_diff is computed per
        group. Top-level parent classes are excluded. The number above each bar shows the
        count of qualifying subclasses.

        **Interpretation:** Identifies which thematic areas benefit most (or least) from
        entity enhancement. Topics with few subclasses should be interpreted cautiously.
    '''),
    'graph7a_entity_type_count': textwrap.dedent('''\
        ### Graph 7a -- Entity type distribution by impact (count)

        **Source:** `entity_impact_all` sheet, columns `entity_type`, `entity_score`.

        **Methodology:** Entities are split into positive, zero, and negative impact
        groups. For each group, the proportion of entity types (person, location,
        organization, general, product, event) is shown as a pie chart. Each entity
        has equal weight. Rare types (idiom, simile, namedphrase, other) are excluded.

        **Interpretation:** Compares whether certain entity types appear more often among
        entities that help, are neutral to, or hurt the classifier.
    '''),
    'graph7b_entity_type_weighted': textwrap.dedent('''\
        ### Graph 7b -- Entity type distribution by impact (score-weighted)

        **Source:** Same as Graph 7a.

        **Methodology:** Same three-way split. Positive and negative groups are weighted
        by absolute entity score; the zero group uses entity counts (since all scores
        are zero).

        **Interpretation:** Highlights which entity types contribute the most total
        positive or negative impact, accounting for the magnitude of each entity's effect.
    '''),
    'graph8_f1_diff_vs_support': textwrap.dedent('''\
        ### Graph 8 -- Absolute F1-score difference vs. class support

        **Source:** `classes_comparison` sheet, column `F1_diff`.

        **Methodology:** Subclasses with support >= 15 are sorted by support ascending.
        A rolling mean with window 50 (min_periods=1) is computed over the absolute value
        |F1_diff| and plotted against the actual support value on a logarithmic x-axis.

        **Interpretation:** Shows how the average magnitude of the F1-score change (regardless
        of direction) evolves as class support grows. Unlike Graph 3b (x = rank), the
        logarithmic support axis reveals the trend directly against class frequency.
    '''),
}


def write_markdown(
    *,
    figures: Sequence[tuple[str, Path, Path]],
    output_dir: Path,
    excel_path: Path,
) -> Path:
    '''Write a detailed Markdown description file for all generated graphs.

    :return: path to the written .md file
    '''
    lines = [
        '# Thesis Graphs -- Detailed Description\n',
        f'**Generated from:** `{excel_path.name}`\n',
        '## Overview\n',
        'This document describes each graph produced by `thesis_graphs.py` for inclusion '
        'in the thesis. All figures compare the entity-enhanced classifier against the '
        'baseline model.\n',
        '## Generated files\n',
        '| Figure | PDF | PNG |',
        '|--------|-----|-----|',
    ]

    for fig_name, pdf, png in figures:
        lines.append(f'| {fig_name} | `{pdf.name}` | `{png.name}` |')

    lines.append('')

    for fig_name, _, _ in figures:
        section = _MD_SECTIONS.get(fig_name)
        if section:
            lines.append(section)

    md_path = output_dir / 'graphs_description.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    LOG.info(f'Wrote markdown documentation {md_path}')
    return md_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all(*, excel_path: Path, output_dir: Path) -> None:
    '''Run all graph generators, write LaTeX snippets and Markdown docs.'''
    _configure_mpl()

    all_figures: list[tuple[str, Path, Path]] = []

    LOG.info('--- Graph 1a/1b: Corpora comparison ---')
    all_figures.extend(plot_corpora(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 2: Language comparison ---')
    all_figures.extend(plot_languages(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 3: F1 diff scatter ---')
    all_figures.extend(plot_f1_diff_scatter(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 3b: F1 diff moving average ---')
    all_figures.extend(plot_f1_diff_moving_avg(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 4: Absolute F1 scatter ---')
    all_figures.extend(plot_f1_abs_scatter(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 4b: Absolute F1 moving average ---')
    all_figures.extend(plot_f1_abs_moving_avg(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 5a/5b: Threshold moving average ---')
    all_figures.extend(plot_threshold_scatter(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 6: IPTC topic comparison ---')
    all_figures.extend(plot_iptc_topic_comparison(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 7a/7b: Entity type pies ---')
    all_figures.extend(plot_entity_type_pies(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Graph 8: F1 diff moving average vs. support ---')
    all_figures.extend(plot_f1_diff_vs_support(excel_path=excel_path, output_dir=output_dir))

    LOG.info('--- Writing LaTeX snippets ---')
    write_latex(figures=all_figures, output_dir=output_dir)

    LOG.info('--- Writing Markdown documentation ---')
    write_markdown(figures=all_figures, output_dir=output_dir, excel_path=excel_path)

    LOG.info(f'All done. {len(all_figures)} figures saved to {output_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    '''Create CLI argument parser.'''
    parser = argparse.ArgumentParser(
        description='Generate thesis graphs from a comparison Excel workbook.',
    )
    parser.add_argument(
        '-i', '--input', required=True, type=Path,
        help='Path to the evaluation comparison Excel file (.xlsx).',
    )
    parser.add_argument(
        '-o', '--output-dir', required=True, type=Path,
        help='Output directory for graphs, LaTeX snippets, and Markdown.',
    )
    return parser


def main() -> None:
    '''Entry point for the thesis graph generator.'''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    args = build_arg_parser().parse_args()

    excel_path: Path = args.input
    if not excel_path.exists():
        LOG.error(f'Input file does not exist: {excel_path}')
        raise SystemExit(1)

    generate_all(excel_path=excel_path, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
