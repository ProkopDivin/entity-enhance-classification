"""Single-knob random seeding for the IPTC entity-enhanced pipeline.

A pipeline-wide :func:`set_global_seed` is the only source of reproducibility.
It seeds Python's :mod:`random`, NumPy, and PyTorch (CPU + CUDA) and switches
cuDNN to deterministic mode (best-effort, not strict). It is called:

1. once at the start of every ClearML component entry point so each component
   process inherits the same RNG state (``run_training_pipeline``, ``run_cv``,
   ``train_best``);
2. before every CV fold's :func:`train_model` so per-fold model
   initialization and DataLoader shuffling are reproducible and decoupled
   from upstream RNG consumption (e.g. early-stopping epoch counts).

The CV fold splitter and Optuna sampler take the same seed directly.
"""

from __future__ import annotations

import logging
import os
import random as _py_random
from typing import Final

import numpy as np

LOGGER = logging.getLogger(__name__)

_SEED_MASK: Final[int] = 0xFFFF_FFFF


def set_global_seed(*, seed: int) -> None:
    """Seed all RNGs the pipeline draws from.

    Seeds Python ``random``, NumPy, and PyTorch (CPU + every visible CUDA
    device), and sets ``torch.backends.cudnn.deterministic = True`` /
    ``benchmark = False``. PyTorch is imported lazily so callers in
    torch-free contexts (e.g. lightweight tooling) can still import this
    module.

    :param seed: Non-negative seed value. Masked to 32 bits where the
        underlying API requires it (NumPy legacy seeders, env vars).
    """
    masked = int(seed) & _SEED_MASK
    _py_random.seed(int(seed))
    np.random.seed(masked)
    os.environ['PYTHONHASHSEED'] = str(masked)
    try:
        import torch
    except ImportError:
        LOGGER.debug('torch not importable; skipping torch seeding')
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fold_seed(*, base_seed: int, fold_idx: int) -> int:
    """Derive a deterministic per-fold seed from the pipeline-wide seed.

    Different folds get different RNG state so model initializations are
    not identical across folds, while still being reproducible.
    """
    return int(base_seed) + int(fold_idx)
