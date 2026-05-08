"""JSON I/O helpers for the assembly (dual-model ensemble) pipeline mode.

Houses the small set of pure read/write helpers used by the assembly path:

- :func:`load_category_list`, :func:`save_category_list`,
  :func:`validate_or_write_cat_list` — shared category list (write-on-first,
  validate-on-rest) so both members agree on the class-to-index mapping.
- :func:`load_thresholds` — per-class threshold map for one member.
- :func:`save_class_to_model_map` — persists the assembly mapping artifact.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


def _atomic_write(*, path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON atomically (tmpfile + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(payload, out, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)
        raise


def load_category_list(*, path: Path) -> list[str] | None:
    """
    Return the parsed category list, or ``None`` if the file does not exist.

    :param path: JSON file path containing a list of category ids.
    :return: Category id list or ``None`` if the file is absent.
    """
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as src:
        data = json.load(src)
    if not isinstance(data, list):
        raise ValueError(f'Expected JSON list at {path}, got {type(data).__name__}')
    return [str(item) for item in data]


def save_category_list(*, path: Path, cat_list: Sequence[str]) -> None:
    """
    Persist the category list to JSON atomically.

    :param path: Destination JSON path.
    :param cat_list: Ordered list of category ids.
    """
    _atomic_write(path=path, payload=list(cat_list))


def validate_or_write_cat_list(
    *,
    path: Path,
    cat_list: Sequence[str],
    member_label: str,
) -> None:
    """
    Write ``cat_list`` on first call, validate equality on subsequent calls.

    The first member to reach this path persists its category list. Later
    members assert their ``corpus.catList`` matches exactly (same elements
    in the same order). Mismatches raise ``ValueError`` with a diff-style
    message so the failure mode is obvious.

    :param path: Shared category list JSON path.
    :param cat_list: This member's ``corpus.catList``.
    :param member_label: Member label used in the error message.
    :raises ValueError: When the existing file disagrees with ``cat_list``.
    """
    existing = load_category_list(path=path)
    incoming = [str(c) for c in cat_list]
    if existing is None:
        save_category_list(path=path, cat_list=incoming)
        LOGGER.info(
            f'Assembly: wrote shared category list path={path} '
            f'n_classes={len(incoming)} member={member_label}'
        )
        return
    if existing == incoming:
        LOGGER.info(
            f'Assembly: category list validated for member={member_label} '
            f'n_classes={len(incoming)}'
        )
        return
    existing_set = set(existing)
    incoming_set = set(incoming)
    missing = sorted(existing_set - incoming_set)
    extra = sorted(incoming_set - existing_set)
    order_change = (
        existing_set == incoming_set
        and existing != incoming
    )
    raise ValueError(
        f'Assembly category list mismatch for member={member_label}: '
        f'missing={missing[:10]}{"..." if len(missing) > 10 else ""} '
        f'extra={extra[:10]}{"..." if len(extra) > 10 else ""} '
        f'order_changed={order_change} path={path}'
    )


def load_thresholds(
    *,
    path: Path,
    cat_list: Sequence[str],
    default_threshold: float,
) -> dict[str, float]:
    """
    Load per-class thresholds and align them with ``cat_list``.

    Categories present in ``cat_list`` but missing from the file fall back to
    ``default_threshold`` with a warning. Categories present in the file but
    not in ``cat_list`` are dropped with a warning.

    :param path: Threshold JSON path, format ``{cat_id: float}``.
    :param cat_list: Reference category list.
    :param default_threshold: Fallback used for missing classes.
    :return: ``{cat_id: float}`` covering exactly ``cat_list``.
    :raises ValueError: When the file is missing or unparseable.
    """
    if not path.exists():
        raise ValueError(f'Threshold file does not exist: {path}')
    with path.open('r', encoding='utf-8') as src:
        raw = json.load(src)
    if not isinstance(raw, Mapping):
        raise ValueError(f'Expected JSON object at {path}, got {type(raw).__name__}')

    parsed: dict[str, float] = {str(k): float(v) for k, v in raw.items()}
    cat_set = set(str(c) for c in cat_list)
    extras = sorted(set(parsed) - cat_set)
    if extras:
        LOGGER.warning(
            f'Threshold file has {len(extras)} extra classes not in cat_list; '
            f'dropping. path={path} sample={extras[:5]}'
        )
    aligned: dict[str, float] = {}
    missing: list[str] = []
    for cat in cat_list:
        cat_id = str(cat)
        if cat_id in parsed:
            aligned[cat_id] = parsed[cat_id]
        else:
            aligned[cat_id] = float(default_threshold)
            missing.append(cat_id)
    if missing:
        LOGGER.warning(
            f'Threshold file missing {len(missing)} classes; using default='
            f'{default_threshold}. path={path} sample={missing[:5]}'
        )
    return aligned


def save_class_to_model_map(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """
    Persist the class-to-model mapping JSON atomically.

    Schema follows the assembly plan: ``schema_version``, ``member_labels``,
    ``primary_index``, ``category_list_path``, ``thresholds_paths``,
    ``assignments``, ``stitched_thresholds``.

    :param path: Output JSON path.
    :param payload: Serializable mapping to persist.
    """
    _atomic_write(path=path, payload=dict(payload))
