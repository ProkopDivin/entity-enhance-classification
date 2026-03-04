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

from iptc_entity_pipeline.clearml_pipeline import run_local_pipeline
from iptc_entity_pipeline.config import PipelineConfig

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
        action='store_true',
        help='Run in local mode with default config resolved from current working directory.',
    )
    return parser


def main() -> None:
    """
    Run pipeline from CLI.

    :return: None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = build_arg_parser().parse_args()

    config = PipelineConfig()
    run_local_pipeline(config=config)
    LOGGER.info('Pipeline execution finished.')


if __name__ == '__main__':
    main()

