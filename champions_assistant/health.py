from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adb import AdbError
from .config import AppConfig, ROI_KEYS
from .data_loader import DataRepository
from .emulator import adb_client_from_config
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
        result = ["环境检查"]
        for item in self.items:
            label = "正常" if item.ok else "警告"
            result.append(f"  [{label}] {_item_label(item.name)}：{_localize_detail(item.name, item.detail)}")
        if self.empty_rois:
            result.append(
                f"  [警告] ROI：{len(self.empty_rois)} 个 ROI 未配置；"
                f"对手预览默认 ROI 生效：{self.opponent_preview_defaults}/6"
            )
        else:
            result.append("  [正常] ROI：已全部配置")
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
            HealthItem("emulator", adb_status.startswith("ok"), f"{config.emulator.profile}: {adb_status}"),
            HealthItem("capture", True, f"method={config.emulator.capture_method}"),
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
    client = adb_client_from_config(config)
    try:
        discovery = client.discover_device()
    except AdbError as exc:
        return f"warn: {exc}"
    online = [device.serial for device in discovery.devices if device.online]
    if discovery.selected_serial:
        detail = discovery.message
        if discovery.attempted_tcp:
            detail += f"; tried tcp: {', '.join(discovery.attempted_tcp)}"
        return f"ok: {detail}"
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


def _item_label(name: str) -> str:
    return {
        "data": "数据",
        "vision": "视觉依赖",
        "ocr": "OCR",
        "templates": "模板",
        "dataset": "数据集",
        "adb": "ADB",
        "emulator": "模拟器",
    }.get(name, name)


def _localize_detail(name: str, detail: str) -> str:
    if name == "vision":
        return "OpenCV/Numpy 可用" if "opencv/numpy available" in detail else "缺少视觉依赖，请运行：python -m pip install -e .[vision]"
    if name == "ocr":
        if detail == "none":
            return "未检测到可用 OCR 引擎"
        if detail.startswith("error:"):
            return "OCR 检查失败：" + detail.removeprefix("error:").strip()
        return f"当前引擎：{detail}"
    if name == "templates":
        parts = detail.split()
        if len(parts) >= 4 and parts[1] == "files" and parts[2] == "across":
            return f"{parts[0]} 个模板文件，覆盖 {parts[3]} 个物种"
    if name == "dataset" and detail.endswith("manifest(s)"):
        return f"{detail.split()[0]} 个 manifest"
    if name in {"adb", "emulator"}:
        return _localize_adb_detail(detail)
    return detail


def _localize_adb_detail(detail: str) -> str:
    prefix = ""
    text = detail
    if ": " in detail:
        maybe_prefix, rest = detail.split(": ", 1)
        if maybe_prefix in {"ok", "warn", "ldplayer9", "generic"}:
            if maybe_prefix in {"ldplayer9", "generic"}:
                prefix = f"{maybe_prefix}："
                text = rest
            else:
                text = rest
    if text.startswith("ok: ") or text.startswith("warn: "):
        text = text.split(": ", 1)[1]
    replacements = (
        ("selected LDPlayer 9 device", "已选择雷电模拟器 9 设备"),
        ("selected online device", "已选择在线设备"),
        ("using configured device_serial", "使用配置的 device_serial"),
        ("tried tcp", "已尝试 TCP"),
        ("ADB executable not found", "找不到 ADB 可执行文件"),
        ("No online ADB device found. Start the emulator and enable ADB.", "未发现在线 ADB 设备，请启动模拟器并启用 ADB"),
        ("no online devices", "未发现在线设备"),
        ("multiple online devices", "发现多个在线设备"),
        ("Multiple online ADB devices found", "发现多个在线 ADB 设备"),
        ("Multiple LDPlayer 9 devices found", "发现多个雷电模拟器 9 设备"),
        ("Set device_serial in config/app.toml.", "请在 config/app.toml 中设置 device_serial。"),
        ("No online LDPlayer 9 ADB device found.", "未发现在线雷电模拟器 9 ADB 设备。"),
        ("Tried emulator-5554/5556/5558/5560 and 127.0.0.1:5555/5557/5559/5561.", "已尝试 emulator-5554/5556/5558/5560 和 127.0.0.1:5555/5557/5559/5561。"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return prefix + text
