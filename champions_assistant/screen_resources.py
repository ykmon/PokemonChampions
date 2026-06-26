from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from .models import Rect
from .paths import DEFAULT_RESOURCES_DIR


@dataclass(frozen=True)
class ScreenResource:
    name: str
    base_width: int
    base_height: int
    recognizer: str
    rois: dict[str, Rect]
    detection: dict[str, Any]
    thresholds: dict[str, float]
    recognizers: dict[str, str]
    template_groups: dict[str, str]


def load_screen_resources(root: Path | str = DEFAULT_RESOURCES_DIR) -> dict[str, ScreenResource]:
    resource_root = Path(root) / "screens"
    if not resource_root.exists():
        return {}
    resources: dict[str, ScreenResource] = {}
    for path in sorted(resource_root.glob("*.toml")):
        resource = _load_screen_resource(path)
        resources[resource.name] = resource
    return resources


def load_resource_rois(root: Path | str = DEFAULT_RESOURCES_DIR) -> dict[str, Rect]:
    rois: dict[str, Rect] = {}
    for resource in load_screen_resources(root).values():
        rois.update(resource.rois)
    return rois


def _load_screen_resource(path: Path) -> ScreenResource:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    screen = raw.get("screen", {})
    roi_raw = raw.get("roi", {})
    recognizer = str(screen.get("recognizer", "") or "")
    return ScreenResource(
        name=str(screen.get("name", path.stem) or path.stem),
        base_width=int(screen.get("base_width", 1920) or 1920),
        base_height=int(screen.get("base_height", 1080) or 1080),
        recognizer=recognizer,
        rois={str(key): Rect.from_mapping(value if isinstance(value, dict) else {}) for key, value in roi_raw.items()},
        detection=_string_keyed_mapping(raw.get("detection", {})),
        thresholds=_float_mapping(raw.get("thresholds", {})),
        recognizers=_resource_string_mapping(raw, "recognizers", "recognizer", default=recognizer),
        template_groups=_resource_string_mapping(raw, "template_groups", "template_group"),
    )


def _string_keyed_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _float_mapping(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in raw.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _resource_string_mapping(raw: dict[str, Any], flat_key: str, nested_key: str, *, default: str = "") -> dict[str, str]:
    values: dict[str, str] = {}
    flat = raw.get(flat_key, {})
    if isinstance(flat, dict):
        for key, value in flat.items():
            if isinstance(value, dict):
                item = value.get("type", value.get("name", default))
            else:
                item = value
            item_text = str(item or "").strip()
            if item_text:
                values[str(key)] = item_text

    nested = raw.get(nested_key, {})
    if isinstance(nested, dict):
        for key, value in nested.items():
            if not isinstance(value, dict):
                continue
            item_text = str(value.get("type", value.get("name", default)) or "").strip()
            if item_text:
                values[str(key)] = item_text
    return values
