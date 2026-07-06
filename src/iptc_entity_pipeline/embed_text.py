'''
Embed text using Jina embedding models.

Variants:
- jina-v3 (default): retrieval.passage / retrieval.query
- jina-v3-classification
- jina-v5: retrieval query/document (same roles as jina-v3)
- jina-v5-classification
'''

import argparse
import json
import sys
from pathlib import Path

from geneea.core import logutil

from iptc_entity_pipeline.jina_embed import JinaModelVariant, embed_query, embed_texts, setup_model

LOG = logutil.getLogger(__package__, __file__)


def read_texts(*, path: Path | None) -> list[str]:
    '''
    Read non-empty lines from a file or stdin.

    :param path: Input file path; reads stdin when ``None``
    :return: Text lines
    '''
    if path is None:
        return [line.rstrip('\n') for line in sys.stdin if line.strip()]
    return [
        line.rstrip('\n')
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def main() -> int:
    '''
    Run CLI entry point.

    :return: Process return code
    '''
    variants = [variant.value for variant in JinaModelVariant]
    argparser = argparse.ArgumentParser(description='Embed text with Jina models')
    argparser.add_argument('-i', '--input', type=Path, help='Input file (one text per line); default: stdin')
    argparser.add_argument('-o', '--output', type=Path, help='Output JSON file; default: stdout')
    argparser.add_argument(
        '--variant',
        choices=variants,
        default=JinaModelVariant.JINA_V3.value,
        help='Model variant (default: jina-v3)',
    )
    argparser.add_argument(
        '--task',
        choices=['passage', 'query', 'classification'],
        default='passage',
        help='Embedding task: passage (documents), query (search), classification (default: passage)',
    )
    argparser.add_argument('--dim', type=int, default=None, help='Embedding dimension (default: 1024)')
    argparser.add_argument('--setup-only', action='store_true', help='Only download/load model, no embedding')

    logutil.addLogArguments(argparser)
    args = argparser.parse_args()
    logutil.configureFromArgs(args)

    setup_model(args.variant)
    if args.setup_only:
        LOG.info(f'Model ready, variant={args.variant}')
        return 0

    texts = read_texts(path=args.input)
    if args.task == 'query':
        if len(texts) != 1:
            argparser.error('Query task expects exactly one input line')
        embeddings = [embed_query(texts[0], variant=args.variant, embedding_dim=args.dim)]
    else:
        embeddings = embed_texts(
            texts,
            variant=args.variant,
            task_kind=args.task,
            embedding_dim=args.dim,
        )

    result = {
        'variant': args.variant,
        'task': args.task,
        'embeddings': embeddings,
    }

    out = json.dumps(result, ensure_ascii=False)
    if args.output:
        args.output.write_text(out, encoding='utf-8')
    else:
        sys.stdout.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
