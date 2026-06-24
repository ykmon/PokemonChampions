from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adb import AdbClient, AdbError
from .config import AppConfig, ROI_KEYS
from .data_loader import DataRepository
from .evaluation import find_manifest_paths
from .ocr import OptionalOcrEngine
from .paths import PROJECT_ROOT
from .roi import VisionDependencyError
from .templates import _cv2, _np, default_opponent_preview_rois_1920


@dataclass(frozen=True)
class HealthItem:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class HealthReport:
    items: tuple[HealthItem, ...]
    empty_rois: tuple[str, ...]
    opponent_preview_defaults: int

    @property
    def blocking_ok(self) -> bool:
        required = {"data", "vision", "templates"}
        return all(item.ok for item in self.items if item.name in required)

    @property
    def warnings(self) -> int:
        return sum(1 for item in self.items if not item.ok) + (1 if self.empty_rois else 0)

    def lines(self) -> list[str]:
        result = ["Health check"]
        for item in self.items:
            label = "ok" if item.ok else "warn"
            result.append(f"  [{label}] {item.name}: {item.detail}")
        if self.empty_rois:
            result.append(
                f"  [warn] roi: {len(self.empty_rois)} empty ROI(s); "
                f"opponent preview defaults active: {self.opponent_preview_defaults}/6"
            )
        else:
            result.append("  [ok] roi: all configured")
        return result


def build_health_report(config: AppConfig, *, project_root: Path = PROJECT_ROOT) -> HealthReport:
    repository_ok = _check_data_repository(config)
    vision_ok = _check_vision()
    ocr_status = _check_ocr()
    template_count, template_species = _count_templates(project_root)
    dataset_manifests = find_manifest_paths(project_root / "dataset", "all")
    empty_rois = tuple(key for key in ROI_KEYS if not config.rois.get(key) or not config.rois[key].enabled)
    default_preview_rois = default_opponent_preview_rois_1920()
    preview_defaults = sum(1 for key in default_preview_rois if key not in empty_rois)
    adb_status = _check_adb(config)

    return HealthReport(
        items=(
            HealthItem("data", repository_ok, f"{config.data_dir}"),
            HealthItem("vision", vision_ok, "opencv/numpy available" if vision_ok else "install with: python -m pip install -e .[vision]"),
            HealthItem("ocr", ocr_status != "none", ocr_status),
            HealthItem("templates", template_count > 0, f"{template_count} files across {template_species} species"),
            HealthItem("dataset", bool(dataset_manifests), f"{len(dataset_manifests)} manifest(s)"),
            HealthItem("adb", adb_status.startswith("ok"), adb_status),
        ),
        empty_rois=empty_rois,
        opponent_preview_defaults=preview_defaults,
    )


def _check_data_repository(config: AppConfig) -> bool:
    try:
        repository = DataRepository(config.data_dir)
    except (OSError, KeyError, ValueError):
        return False
    return bool(repository.pokemon_by_id and repository.moves_by_name)


def _check_vision() -> bool:
    try:
        _cv2()
        _np()
    except VisionDependencyError:
        return False
    return True


def _check_ocr() -> str:
    try:
        return OptionalOcrEngine().engine_name
    except Exception as exc:
        return f"error: {exc}"


def _check_adb(config: AppConfig) -> str:
    client = AdbClient(config.adb_path, config.device_serial)
    try:
        devices = client.list_devices()
    except AdbError as exc:
        return f"warn: {exc}"
    online = [device.serial for device in devices if device.online]
    if not online:
        return "warn: no online devices"
    if len(online) > 1 and not config.device_serial:
        return f"warn: multiple online devices: {', '.join(online)}"
    return f"ok: {', '.join(online)}"


def _count_templates(project_root: Path) -> tuple[int, int]:
    root = project_root / "assets" / "pokemon_templates"
    if not root.exists():
        return 0, 0
    species_dirs = [path for path in root.iterdir() if path.is_dir()]
    count = sum(1 for species_dir in species_dirs for _ in species_dir.glob("*.png"))
    return count, len(species_dirs)
