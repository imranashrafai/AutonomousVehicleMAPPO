"""Portable paths and runtime configuration for the dashboard applications."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAPPO_ROOT = PROJECT_ROOT / "MAPPO_AVs" / "mappo"


def _path_from_env(variable: str, default: Path) -> Path:
    value = os.getenv(variable)
    path = Path(value).expanduser() if value else default
    return path.resolve()


RESULTS_DIR_ROOT = _path_from_env("AV_MAPPO_RESULTS_DIR", MAPPO_ROOT / "results")
CHECKPOINT_BASE = RESULTS_DIR_ROOT / "MyEnv" / "Intersection_MAPPO"
TRAIN_SCRIPT = _path_from_env("AV_MAPPO_TRAIN_SCRIPT", MAPPO_ROOT / "train" / "train.py")
PYTHON_EXE = os.getenv("AV_MAPPO_PYTHON", sys.executable)

LOCAL_DATA_DIR = _path_from_env("AV_MAPPO_DATA_DIR", PROJECT_ROOT / ".local_data")
SCREENSHOT_ROOT = _path_from_env("AV_MAPPO_SCREENSHOT_DIR", LOCAL_DATA_DIR / "screenshots")
EVALUATION_OUTPUT = _path_from_env("AV_MAPPO_EVALUATION_DIR", LOCAL_DATA_DIR / "evaluations")
USERS_FILE = _path_from_env("AV_MAPPO_USERS_FILE", LOCAL_DATA_DIR / "users.json")
SUPPORT_FILE = _path_from_env("AV_MAPPO_SUPPORT_FILE", LOCAL_DATA_DIR / "support_tickets.json")
LEGACY_USERS_FILE = PROJECT_ROOT / "users.json"

WATERMARK_TEXT_DEFAULT = os.getenv(
    "AV_MAPPO_WATERMARK",
    "Autonomous Vehicle MAPPO",
)


def ensure_runtime_directories() -> None:
    """Create directories used for generated local data."""
    for directory in (LOCAL_DATA_DIR, SCREENSHOT_ROOT, EVALUATION_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
