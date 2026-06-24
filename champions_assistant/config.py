from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from .models import BattleFormat, Rect
from .paths import DEFAULT_CONFIG_PATH, DEFAULT_DATA_DIR, DEFAULT_SCREENSHOTS_DIR


LEGACY_ROI_KEYS = ("self_name", "opponent_name", "self_hp", "opponent_hp", "turn")
TEAM_ROI_KEYS = tuple(
    f"{side}_preview_{index}"
    for side in ("player", "opponent")
    for index in range(1, 7)
)
ACTIVE_ROI_KEYS = tuple(
    f"{side}_active_{index}"
    for side in ("player", "opponent")
    for index in range(1, 3)
)
ACTIVE_HP_ROI_KEYS = tuple(
    f"{side}_active_{index}_hp"
    for side in ("player", "opponent")
    for index in range(1, 3)
)
ROI_KEYS = tuple(dict.fromkeys((*TEAM_ROI_KEYS, *ACTIVE_ROI_KEYS, *ACTIVE_HP_ROI_KEYS, "turn", *LEGACY_ROI_KEYS)))

DEFAULT_ROIS_1920 = {
    f"opponent_preview_{index}": Rect(x=1375, y=144 + (index - 1) * 130, width=150, height=115)
    for index in range(1, 7)
}


@dataclass
class UiConfig:
    always_on_top: bool = True
    compact: bool = False


@dataclass
class AppConfig:
    adb_path: str = "adb"
    device_serial: str = ""
    capture_interval_ms: int = 1200
    language: str = "zh"
    default_format: BattleFormat = BattleFormat.SINGLES_63
    data_dir: Path = DEFAULT_DATA_DIR
    screenshots_dir: Path = DEFAULT_SCREENSHOTS_DIR
    rois: dict[str, Rect] = field(default_factory=lambda: {key: Rect() for key in ROI_KEYS})
    ui: UiConfig = field(default_factory=UiConfig)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent
    data_dir = _resolve_relative(base_dir, raw.get("data_dir", "data"))
    screenshots_dir = _resolve_relative(base_dir, raw.get("screenshots_dir", "screenshots"))
    roi_raw = raw.get("roi", {})
    rois = _load_rois(roi_raw)
    ui_raw = raw.get("ui", {})

    return AppConfig(
        adb_path=str(raw.get("adb_path", "adb") or "adb"),
        device_serial=str(raw.get("device_serial", "") or ""),
        capture_interval_ms=int(raw.get("capture_interval_ms", 1200) or 1200),
        language=str(raw.get("language", "zh") or "zh"),
        default_format=_parse_default_format(raw.get("default_format", BattleFormat.SINGLES_63.value)),
        data_dir=data_dir,
        screenshots_dir=screenshots_dir,
        rois=rois,
        ui=UiConfig(
            always_on_top=bool(ui_raw.get("always_on_top", True)),
            compact=bool(ui_raw.get("compact", False)),
        ),
    )


def save_config(config: AppConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_to_toml(config, config_path), encoding="utf-8")


def _resolve_relative(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _path_to_config_value(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _to_toml(config: AppConfig, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    base_dir = config_path.parent.parent
    lines = [
        f'adb_path = "{_escape(config.adb_path)}"',
        f'device_serial = "{_escape(config.device_serial)}"',
        f"capture_interval_ms = {int(config.capture_interval_ms)}",
        f'language = "{_escape(config.language)}"',
        f'default_format = "{_escape(config.default_format.value)}"',
        f'data_dir = "{_escape(_path_to_config_value(config.data_dir, base_dir))}"',
        f'screenshots_dir = "{_escape(_path_to_config_value(config.screenshots_dir, base_dir))}"',
        "",
        "[ui]",
        f"always_on_top = {_bool(config.ui.always_on_top)}",
        f"compact = {_bool(config.ui.compact)}",
        "",
    ]
    for key in ROI_KEYS:
        rect = config.rois.get(key, Rect())
        lines.extend(
            [
                f"[roi.{key}]",
                f"x = {rect.x}",
                f"y = {rect.y}",
                f"width = {rect.width}",
                f"height = {rect.height}",
                "",
            ]
        )
    return "\n".join(lines)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parse_default_format(value: Any) -> BattleFormat:
    try:
        return BattleFormat.parse(str(value))
    except ValueError:
        return BattleFormat.SINGLES_63


def _load_rois(roi_raw: dict[str, Any]) -> dict[str, Rect]:
    rois = {key: Rect.from_mapping(roi_raw.get(key, {})) for key in ROI_KEYS}
    for key, rect in DEFAULT_ROIS_1920.items():
        if not rois[key].enabled:
            rois[key] = rect
    legacy_mappings = {
        "self_name": "player_active_1",
        "opponent_name": "opponent_active_1",
        "self_hp": "player_active_1_hp",
        "opponent_hp": "opponent_active_1_hp",
    }
    for legacy_key, new_key in legacy_mappings.items():
        legacy_rect = Rect.from_mapping(roi_raw.get(legacy_key, {}))
        if legacy_rect.enabled and not rois[new_key].enabled:
            rois[new_key] = legacy_rect
    return rois
