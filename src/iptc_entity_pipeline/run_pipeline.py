"""
CLI entrypoint for running the IPTC entity-enhanced ClearML pipeline (v1).
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow direct script execution:
# python src/iptc_entity_pipeline/run_pipeline.py
if __package__ is None or __package__ == '':
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from iptc_entity_pipeline.clearml_pipeline import run_pipeline
from iptc_entity_pipeline.config import get_config, list_config_names

LOGGER = logging.getLogger(__name__)

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create CLI parser for pipeline execution.

    :return: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description='Train and evaluate entity-enhanced IPTC classifier with ClearML.',
    )
    parser.add_argument(
        '--task-name',
        default='iptc-entity-enhanced-v1',
        help='ClearML task name.',
    )
    parser.add_argument(
        '--local',
        '-l',
        action='store_true',
        help='Run in local mode with default config resolved from current working directory.',
    )
    parser.add_argument(
        '--config',
        '-c',
        default='wpentities',
        choices=list_config_names(),
        help='Config variant to run (e.g. "wpentities", "article_only").',
    )
    return parser


def main() -> None:
    """
    Run pipeline from CLI.

    :return: None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = build_arg_parser().parse_args()
    config_name = args.config
    config = get_config(config_name=config_name)
    LOGGER.info('Using config: %s', config_name)
    run_pipeline(config=config, config_name=config_name, is_local=args.local)
    LOGGER.info('Pipeline execution finished.')


if __name__ == '__main__':
    main()

