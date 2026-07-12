"""Shared loaders for category-id subsets defined in YAML resource files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import yaml

RESOURCES_DIR = Path(__file__).resolve().parent / 'resources'
RELEVANT_CATS_PATH = RESOURCES_DIR / 'relevant_cats.yaml'
TAIL_CATS_PATH = RESOURCES_DIR / 'tail_cats.yaml'
ZERO_CATS_PATH = RESOURCES_DIR / 'zero-cats.yaml'


def load_category_ids_from_yaml(*, path: Path) -> set[str]:
    """
    Load category ids from a YAML file under ``categories: [{id: ...}, ...]``.

    :param path: Path to the category YAML file.
    :return: Set of category ids.
    """
    if not path.is_file():
        raise FileNotFoundError(f'Category YAML file not found: {path}')
    with open(path, encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f'Invalid category YAML shape (expected mapping): {path}')
    categories = payload.get('categories')
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise ValueError(f'Invalid category YAML shape (expected list under "categories"): {path}')
    cat_ids: set[str] = set()
    for item in categories:
        if not isinstance(item, Mapping):
            raise ValueError(f'Invalid category YAML item (expected mapping): {path}')
        cat_id = str(item.get('id', '')).strip()
        if not cat_id:
            raise ValueError(f'Category entry without id in: {path}')
        cat_ids.add(cat_id)
    if not cat_ids:
        raise ValueError(f'No category ids found in: {path}')
    return cat_ids


@lru_cache(maxsize=1)
def load_relevant_cat_ids() -> set[str]:
    """Load relevant category ids from ``resources/relevant_cats.yaml``."""
    return load_category_ids_from_yaml(path=RELEVANT_CATS_PATH)


@lru_cache(maxsize=1)
def load_tail_cat_ids() -> set[str]:
    """Load tail category ids from ``resources/tail_cats.yaml``."""
    return load_category_ids_from_yaml(path=TAIL_CATS_PATH)


@lru_cache(maxsize=1)
def load_zero_cat_ids() -> set[str]:
    """Load zero-support category ids from ``resources/zero-cats.yaml``."""
    return load_category_ids_from_yaml(path=ZERO_CATS_PATH)
