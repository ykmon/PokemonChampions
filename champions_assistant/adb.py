from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    status: str
    details: str = ""

    @property
    def online(self) -> bool:
        return self.status == "device"


class AdbClient:
    def __init__(self, adb_path: str = "adb", serial: str = "") -> None:
        self.adb_path = adb_path
        self.serial = serial

    def list_devices(self) -> list[DeviceInfo]:
        result = self._run(["devices", "-l"], text=True)
        return parse_devices(result.stdout)

    def selected_serial(self) -> str:
        if self.serial:
            return self.serial
        online_devices = [device for device in self.list_devices() if device.online]
        if not online_devices:
            raise AdbError("No online ADB device found. Start LDPlayer and enable ADB.")
        if len(online_devices) > 1:
            serials = ", ".join(device.serial for device in online_devices)
            raise AdbError(f"Multiple online ADB devices found: {serials}. Set device_serial in config/app.toml.")
        return online_devices[0].serial

    def capture_screenshot(self, out_path: Path | str | None = None, timeout: float = 5) -> bytes:
        serial = self.selected_serial()
        args = ["-s", serial, "exec-out", "screencap", "-p"]
        result = self._run(args, text=False, timeout=timeout)
        image_bytes = result.stdout.replace(b"\r\n", b"\n")
        if not image_bytes.startswith(b"\x89PNG"):
            raise AdbError("ADB screencap did not return a PNG image.")
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)
        return image_bytes

    def _run(
        self,
        args: list[str],
        *,
        text: bool,
        timeout: float = 5,
    ) -> subprocess.CompletedProcess:
        command = [self.adb_path, *args]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=text,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"ADB executable not found: {self.adb_path}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB command timed out: {' '.join(command)}") from exc

        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="ignore")
            raise AdbError(stderr.strip() or f"ADB command failed: {' '.join(command)}")
        return result


def parse_devices(output: str) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            details = parts[2] if len(parts) > 2 else ""
            devices.append(DeviceInfo(serial=parts[0], status=parts[1], details=details))
    return devices
