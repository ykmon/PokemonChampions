from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app.toml"
DEFAULT_VISION_CONFIG_PATH = PROJECT_ROOT / "config" / "vision.toml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
DEFAULT_RESOURCES_DIR = PROJECT_ROOT / "resources"
