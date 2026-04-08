#!/usr/bin/env python3
"""
Generate word-cloud visualisations of the most common entity stdForms.

Reads article_2_entities.tsv, splits entities into two groups (with wdId
and without wdId), counts stdForm occurrences, and renders a word cloud
for each group.
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib  # type: ignore[import-not-found]
matplotlib.use('Agg')  # noqa: E402 - must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
from wordcloud import WordCloud  # type: ignore[import-not-found]  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from iptc_entity_pipeline.data_loading import load_article_entities  # noqa: E402

LOG = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
ARTICLE_ENTITIES_TSV = 'article_2_entities.tsv'
OUTPUT_DIR = DATA_DIR / 'origin-corpora' / 'graphs'


def count_stdforms(
    entities_by_article: dict[str, list[dict]],
) -> tuple[Counter[str], Counter[str]]:
    """
    Split entities into linked (with wdId) / unlinked (without wdId)
    and count stdForm occurrences in each group.

    :param entities_by_article: mapping article_id -> list of entity dicts
    :return: (with_wdid_counts, without_wdid_counts)
    """
    with_wdid: Counter[str] = Counter()
    without_wdid: Counter[str] = Counter()

    for entities in entities_by_article.values():
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            std_form = ent.get('stdForm', '').strip()
            if not std_form:
                continue
            if ent.get('wdId'):
                with_wdid[std_form] += 1
            else:
                without_wdid[std_form] += 1

    return with_wdid, without_wdid


def render_wordcloud(
    frequencies: dict[str, int],
    *,
    title: str,
    output_path: Path,
    top_n: int = 200,
) -> None:
    """
    Render a word-cloud image from frequency counts.

    :param frequencies: mapping stdForm -> count
    :param title: plot title
    :param output_path: path to save the PNG
    :param top_n: keep only top-N most frequent terms
    """
    top_items = dict(Counter(frequencies).most_common(top_n))
    if not top_items:
        LOG.warning('No data for word cloud "%s", skipping.', title)
        return

    wc = WordCloud(
        width=1600,
        height=900,
        background_color='white',
        max_words=top_n,
        colormap='viridis',
        prefer_horizontal=0.7,
        min_font_size=8,
    ).generate_from_frequencies(top_items)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=20, pad=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    LOG.info('Saved word cloud to %s (%d unique terms)', output_path, len(top_items))


def main() -> None:
    argparser = argparse.ArgumentParser(
        description='Generate word-cloud visualisations of entity stdForms (with/without wdId).',
    )
    argparser.add_argument(
        '-n', '--top-n', type=int, default=200,
        help='Number of most common stdForms to include in each word cloud (default: 200)',
    )
    argparser.add_argument(
        '-o', '--output-dir', type=str, default=None,
        help='Directory to save PNG files (default: origin-corpora/graphs/)',
    )
    argparser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose (DEBUG) logging',
    )
    args = argparser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    tsv_path = DATA_DIR / ARTICLE_ENTITIES_TSV
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info('Loading entities from %s', tsv_path)
    entities_by_article = load_article_entities(article_entities_tsv=str(tsv_path))

    with_wdid, without_wdid = count_stdforms(entities_by_article)
    LOG.info(
        'Entities with wdId: %d unique stdForms (%d total occurrences)',
        len(with_wdid), sum(with_wdid.values()),
    )
    LOG.info(
        'Entities without wdId: %d unique stdForms (%d total occurrences)',
        len(without_wdid), sum(without_wdid.values()),
    )

    render_wordcloud(
        with_wdid,
        title='Most common entity stdForms (with wdId)',
        output_path=out_dir / 'wordcloud_with_wdid.png',
        top_n=args.top_n,
    )

    render_wordcloud(
        without_wdid,
        title='Most common entity stdForms (without wdId)',
        output_path=out_dir / 'wordcloud_without_wdid.png',
        top_n=args.top_n,
    )

    LOG.info('Done.')


if __name__ == '__main__':
    main()
