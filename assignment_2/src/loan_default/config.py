"""Configuration loading for the loan-default package."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load the YAML configuration file into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()

# config.yaml stores paths relative to the project root. Resolve them here so
# the pipeline behaves the same whatever the current working directory is
# (Path.__truediv__ leaves an already-absolute override untouched).
CONFIG["data"]["raw_path"] = str(PROJECT_ROOT / CONFIG["data"]["raw_path"])
CONFIG["model"]["artifact_path"] = str(PROJECT_ROOT / CONFIG["model"]["artifact_path"])
