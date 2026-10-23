"""Shared utilities: random seed management, timer, and logging setup."""

import logging
import time
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global NumPy random seed for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value passed to ``numpy.random.seed``.
    """
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    If the logger has no handlers yet a ``StreamHandler`` is attached so that
    messages are visible in interactive sessions without requiring external
    logging configuration.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class Timer:
    """Context manager that logs elapsed wall-clock time.

    Parameters
    ----------
    label:
        Description of the timed block, included in the log message.
    logger:
        Logger to write to.  If ``None`` the root logger is used.

    Examples
    --------
    >>> with Timer("simulation"):
    ...     run_long_computation()
    2024-01-01 12:00:00 | INFO     | ... | simulation finished in 3.14 s
    """

    def __init__(
        self,
        label: str = "block",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.label = label
        self._logger = logger or get_logger(__name__)
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._logger.info("Starting: %s", self.label)
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.perf_counter() - self._start
        self._logger.info("%s finished in %.4f s", self.label, elapsed)

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the context was entered."""
        return time.perf_counter() - self._start
