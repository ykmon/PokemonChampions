from __future__ import annotations

from .adb import AdbClient
from .config import AppConfig


def adb_client_from_config(config: AppConfig) -> AdbClient:
    return AdbClient(
        config.adb_path,
        config.device_serial,
        emulator_profile=config.emulator.profile,
        auto_detect=config.emulator.auto_detect,
        connect_timeout_ms=config.emulator.connect_timeout_ms,
        capture_method=config.emulator.capture_method,
    )
