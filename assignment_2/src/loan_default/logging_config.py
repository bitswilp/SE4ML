"""Centralised logging setup (Objective 1.3).

One helper so every module logs with the same format and levels
(INFO / WARNING / ERROR) instead of using print().
"""

from __future__ import annotations

import logging


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger; safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
