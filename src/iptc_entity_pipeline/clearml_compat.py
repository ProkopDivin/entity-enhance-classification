"""Compatibility layer for optional ClearML dependency."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

try:
    from clearml import Task as _Task
    from clearml import TaskTypes as _TaskTypes
    from clearml.automation.controller import PipelineDecorator as _PipelineDecorator

    CLEARML_AVAILABLE = True
    Task = _Task
    TaskTypes = _TaskTypes
    PipelineDecorator = _PipelineDecorator
except ModuleNotFoundError:
    CLEARML_AVAILABLE = False

    class _FallbackTaskTypes:
        data_processing = 'data_processing'
        training = 'training'
        testing = 'testing'

    class Task:
        """Fallback ClearML Task API used when clearml is unavailable."""

        @staticmethod
        def current_task() -> None:
            """Mimic ClearML's current-task lookup."""
            return None

    class PipelineDecorator:
        """Fallback decorators that execute pipeline functions as plain Python."""

        @staticmethod
        def component(**_: Any) -> Any:
            """No-op component decorator."""

            def deco(func: Any) -> Any:
                return func

            return deco

        @staticmethod
        def pipeline(**_: Any) -> Any:
            """No-op pipeline decorator."""

            def deco(func: Any) -> Any:
                return func

            return deco

        @staticmethod
        def run_locally() -> None:
            """Local execution is already the default in fallback mode."""
            return None

    TaskTypes = _FallbackTaskTypes()


class LocalTaskLogger:
    """ClearML-like logger that degrades to standard logging."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER

    def report_text(self, text: str, print_console: bool = True, **_: Any) -> None:
        """Log text output when ClearML logger is unavailable."""
        if print_console:
            self._logger.info(text)

    def report_scalar(
        self,
        *,
        title: str,
        series: str,
        value: float,
        iteration: int,
        **_: Any,
    ) -> None:
        """Log scalar output as a compact text line."""
        self._logger.info(
            f'[{title}] {series} iter={iteration} value={value:.6f}'
        )

    def report_table(
        self,
        *,
        title: str,
        series: str,
        iteration: int,
        table_plot: Any | None = None,
        **_: Any,
    ) -> None:
        """Log table report metadata."""
        row_cnt = len(table_plot) if hasattr(table_plot, '__len__') else 'n/a'
        self._logger.info(
            f'[{title}] table={series} iter={iteration} rows={row_cnt}'
        )

    def report_scatter2d(
        self,
        *,
        title: str,
        series: str,
        scatter: Any,
        iteration: int,
        **_: Any,
    ) -> None:
        """Log scatter report metadata."""
        points = len(scatter) if hasattr(scatter, '__len__') else 'n/a'
        self._logger.info(
            f'[{title}] scatter={series} iter={iteration} points={points}'
        )


def is_clearml_available() -> bool:
    """Return True when ClearML package is importable."""
    return CLEARML_AVAILABLE


def get_task_logger(*, task: Any, logger: logging.Logger | None = None) -> Any:
    """Return ClearML logger or local fallback logger."""
    if task is None:
        return LocalTaskLogger(logger=logger)
    return task.get_logger()
