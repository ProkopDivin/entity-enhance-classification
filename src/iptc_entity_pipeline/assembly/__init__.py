"""Dual-model assembly (per-class ensemble) pipeline package.

Groups the assembly aggregation logic, the score-level ensemble model, and
the JSON I/O helpers. The public API is re-exported here so callers can use
``from iptc_entity_pipeline.assembly import build_assembly_from_cv`` etc.
"""

from __future__ import annotations

from iptc_entity_pipeline.assembly.aggregation import (
    AssemblyCvResult,
    build_assembly_from_cv,
    build_member_cv_dev_df,
    build_per_class_f1_df,
    build_per_corpora_df,
    build_threshold_report,
    report_assembly_tables,
    select_class_to_model,
    select_class_to_model_sign_test,
    stitch_thresholds,
)
from iptc_entity_pipeline.assembly.io import load_thresholds, save_class_to_model_map
from iptc_entity_pipeline.assembly.model import AssemblyModel, ClassToModelMap

__all__ = [
    'AssemblyCvResult',
    'AssemblyModel',
    'ClassToModelMap',
    'build_assembly_from_cv',
    'build_member_cv_dev_df',
    'build_per_class_f1_df',
    'build_per_corpora_df',
    'build_threshold_report',
    'load_thresholds',
    'report_assembly_tables',
    'save_class_to_model_map',
    'select_class_to_model',
    'select_class_to_model_sign_test',
    'stitch_thresholds',
]
