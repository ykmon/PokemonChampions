from __future__ import annotations

import gzip
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter

from .fast_preview import VisionFrame


class AdbError(RuntimeError):
    pass


LDPLAYER9_EMULATOR_SERIALS = ("emulator-5554", "emulator-5556", "emulator-5558", "emulator-5560")
LDPLAYER9_TCP_SERIALS = ("127.0.0.1:5555", "127.0.0.1:5557", "127.0.0.1:5559", "127.0.0.1:5561")
LDPLAYER9_SERIALS = (*LDPLAYER9_EMULATOR_SERIALS, *LDPLAYER9_TCP_SERIALS)


class EmulatorProfile(str, Enum):
    LDPLAYER9 = "ldplayer9"
    GENERIC = "generic"

    @classmethod
    def parse(cls, value: str | "EmulatorProfile") -> "EmulatorProfile":
        if isinstance(value, EmulatorProfile):
            return value
        normalized = value.strip().lower().replace("-", "").replace("_", "")
        if normalized in {"ldplayer", "ldplayer9", "leidian", "leidian9"}:
            return cls.LDPLAYER9
        return cls.GENERIC


class CaptureMethod(str, Enum):
    AUTO = "auto"
    RAW = "raw"
    RAW_GZIP = "raw_gzip"
    PNG = "png"

    @classmethod
    def parse(cls, value: str | "CaptureMethod") -> "CaptureMethod":
        if isinstance(value, CaptureMethod):
            return value
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "encode": cls.PNG,
            "encoded": cls.PNG,
            "screencap_png": cls.PNG,
            "rawbygzip": cls.RAW_GZIP,
            "raw_with_gzip": cls.RAW_GZIP,
            "rawgzip": cls.RAW_GZIP,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.AUTO


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    status: str
    details: str = ""

    @property
    def online(self) -> bool:
        return self.status == "device"


@dataclass(frozen=True)
class DeviceDiscovery:
    selected_serial: str | None
    devices: tuple[DeviceInfo, ...]
    attempted_tcp: tuple[str, ...]
    connected_tcp: tuple[str, ...]
    profile: EmulatorProfile
    message: str

    @property
    def ok(self) -> bool:
        return self.selected_serial is not None


@dataclass(frozen=True)
class CaptureAttempt:
    method: CaptureMethod
    ok: bool
    elapsed_ms: float
    message: str = ""


@dataclass(frozen=True)
class CaptureBenchmark:
    selected_method: CaptureMethod | None
    attempts: tuple[CaptureAttempt, ...]

    @property
    def ok(self) -> bool:
        return self.selected_method is not None

    def summary(self) -> str:
        parts = []
        for attempt in self.attempts:
            if attempt.ok:
                parts.append(f"{attempt.method.value}={attempt.elapsed_ms:.1f}ms")
            else:
                parts.append(f"{attempt.method.value}=fail({attempt.message})")
        selected = self.selected_method.value if self.selected_method else "none"
        return f"selected={selected}; " + ", ".join(parts)


@dataclass(frozen=True)
class CapturedFrame:
    frame: VisionFrame
    method: CaptureMethod
    capture_ms: float
    raw_bytes: bytes = b""

    def to_png_bytes(self) -> bytes:
        if self.method == CaptureMethod.PNG and self.raw_bytes.startswith(b"\x89PNG"):
            return self.raw_bytes
        return self.frame.to_png_bytes()


class AdbClient:
    def __init__(
        self,
        adb_path: str = "adb",
        serial: str = "",
        *,
        emulator_profile: str | EmulatorProfile = EmulatorProfile.LDPLAYER9,
        auto_detect: bool = True,
        connect_timeout_ms: int = 1200,
        capture_method: str | CaptureMethod = CaptureMethod.AUTO,
    ) -> None:
        self.adb_path = adb_path
        self.serial = serial
        self.emulator_profile = EmulatorProfile.parse(emulator_profile)
        self.auto_detect = auto_detect
        self.connect_timeout_ms = connect_timeout_ms
        self.capture_method = CaptureMethod.parse(capture_method)
        self._selected_capture_method: CaptureMethod | None = None
        self._capture_benchmark: CaptureBenchmark | None = None

    def list_devices(self) -> list[DeviceInfo]:
        result = self._run(["devices", "-l"], text=True)
        return parse_devices(result.stdout)

    def discover_device(self) -> DeviceDiscovery:
        if self.serial:
            attempted_tcp: tuple[str, ...] = ()
            connected_tcp: tuple[str, ...] = ()
            devices = tuple(self.list_devices())
            matched = next((device for device in devices if device.serial == self.serial), None)
            if matched is None:
                if ":" in self.serial:
                    attempted_tcp = (self.serial,)
                    if self._try_connect(self.serial):
                        connected_tcp = (self.serial,)
                    devices = tuple(self.list_devices())
                    matched = next((device for device in devices if device.serial == self.serial), None)
                if matched is None:
                    raise AdbError(f"Configured device_serial was not found: {self.serial}")
            if not matched.online:
                raise AdbError(f"Configured device_serial is not online: {self.serial} ({matched.status})")
            return DeviceDiscovery(
                selected_serial=self.serial,
                devices=devices,
                attempted_tcp=attempted_tcp,
                connected_tcp=connected_tcp,
                profile=self.emulator_profile,
                message=f"using configured device_serial: {self.serial}",
            )

        devices = tuple(self.list_devices())
        if self.emulator_profile == EmulatorProfile.LDPLAYER9 and self.auto_detect:
            discovery = self._discover_ldplayer9(devices)
            if discovery.ok:
                return discovery
            raise AdbError(discovery.message)

        online_devices = [device for device in devices if device.online]
        if not online_devices:
            raise AdbError("No online ADB device found. Start the emulator and enable ADB.")
        if len(online_devices) > 1:
            serials = ", ".join(device.serial for device in online_devices)
            raise AdbError(f"Multiple online ADB devices found: {serials}. Set device_serial in config/app.toml.")
        return DeviceDiscovery(
            selected_serial=online_devices[0].serial,
            devices=devices,
            attempted_tcp=(),
            connected_tcp=(),
            profile=self.emulator_profile,
            message=f"selected online device: {online_devices[0].serial}",
        )

    def selected_serial(self) -> str:
        discovery = self.discover_device()
        if not discovery.selected_serial:
            raise AdbError(discovery.message)
        return discovery.selected_serial

    def capture_screenshot(self, out_path: Path | str | None = None, timeout: float = 5) -> bytes:
        captured = self.capture_frame(timeout=timeout)
        image_bytes = captured.to_png_bytes()
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)
        return image_bytes

    def capture_frame(self, timeout: float = 5) -> CapturedFrame:
        method = self._resolve_capture_method(timeout=timeout)
        return self._capture_frame_with_method(method, timeout=timeout)

    def benchmark_capture_methods(self, timeout: float = 5) -> CaptureBenchmark:
        if self.capture_method != CaptureMethod.AUTO:
            try:
                captured = self._capture_frame_with_method(self.capture_method, timeout=timeout)
            except (AdbError, ValueError) as exc:
                benchmark = CaptureBenchmark(
                    selected_method=None,
                    attempts=(CaptureAttempt(self.capture_method, False, 0.0, str(exc)),),
                )
                self._capture_benchmark = benchmark
                return benchmark
            benchmark = CaptureBenchmark(
                selected_method=captured.method,
                attempts=(CaptureAttempt(captured.method, True, captured.capture_ms),),
            )
            self._selected_capture_method = captured.method
            self._capture_benchmark = benchmark
            return benchmark

        attempts: list[CaptureAttempt] = []
        successful: list[CaptureAttempt] = []
        for method in (CaptureMethod.RAW, CaptureMethod.RAW_GZIP, CaptureMethod.PNG):
            try:
                captured = self._capture_frame_with_method(method, timeout=timeout)
            except (AdbError, ValueError) as exc:
                attempts.append(CaptureAttempt(method, False, 0.0, _short_error(exc)))
                continue
            attempt = CaptureAttempt(method, True, captured.capture_ms)
            attempts.append(attempt)
            successful.append(attempt)
        selected = min(successful, key=lambda item: item.elapsed_ms).method if successful else None
        benchmark = CaptureBenchmark(selected_method=selected, attempts=tuple(attempts))
        self._selected_capture_method = selected
        self._capture_benchmark = benchmark
        return benchmark

    @property
    def capture_benchmark(self) -> CaptureBenchmark | None:
        return self._capture_benchmark

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

    def _discover_ldplayer9(self, devices: tuple[DeviceInfo, ...]) -> DeviceDiscovery:
        online_ld = [device for device in devices if device.online and device.serial in LDPLAYER9_SERIALS]
        if len(online_ld) == 1:
            return DeviceDiscovery(
                selected_serial=online_ld[0].serial,
                devices=devices,
                attempted_tcp=(),
                connected_tcp=(),
                profile=self.emulator_profile,
                message=f"selected LDPlayer 9 device: {online_ld[0].serial}",
            )
        if len(online_ld) > 1:
            serials = ", ".join(device.serial for device in online_ld)
            return DeviceDiscovery(
                selected_serial=None,
                devices=devices,
                attempted_tcp=(),
                connected_tcp=(),
                profile=self.emulator_profile,
                message=f"Multiple LDPlayer 9 devices found: {serials}. Set device_serial in config/app.toml.",
            )

        connected = []
        attempted = []
        for serial in LDPLAYER9_TCP_SERIALS:
            attempted.append(serial)
            if self._try_connect(serial):
                connected.append(serial)

        refreshed = tuple(self.list_devices())
        online_ld = [device for device in refreshed if device.online and device.serial in LDPLAYER9_SERIALS]
        if len(online_ld) == 1:
            return DeviceDiscovery(
                selected_serial=online_ld[0].serial,
                devices=refreshed,
                attempted_tcp=tuple(attempted),
                connected_tcp=tuple(connected),
                profile=self.emulator_profile,
                message=f"selected LDPlayer 9 device: {online_ld[0].serial}",
            )
        if len(online_ld) > 1:
            serials = ", ".join(device.serial for device in online_ld)
            return DeviceDiscovery(
                selected_serial=None,
                devices=refreshed,
                attempted_tcp=tuple(attempted),
                connected_tcp=tuple(connected),
                profile=self.emulator_profile,
                message=f"Multiple LDPlayer 9 devices found: {serials}. Set device_serial in config/app.toml.",
            )

        return DeviceDiscovery(
            selected_serial=None,
            devices=refreshed,
            attempted_tcp=tuple(attempted),
            connected_tcp=tuple(connected),
            profile=self.emulator_profile,
            message=(
                "No online LDPlayer 9 ADB device found. Tried emulator-5554/5556/5558/5560 "
                "and 127.0.0.1:5555/5557/5559/5561."
            ),
        )

    def _try_connect(self, serial: str) -> bool:
        try:
            result = self._run(
                ["connect", serial],
                text=True,
                timeout=max(0.2, self.connect_timeout_ms / 1000),
            )
        except AdbError:
            return False
        output = f"{result.stdout}\n{result.stderr or ''}".lower()
        return "connected" in output or "already connected" in output

    def _resolve_capture_method(self, *, timeout: float) -> CaptureMethod:
        if self.capture_method != CaptureMethod.AUTO:
            return self.capture_method
        if self._selected_capture_method is None:
            benchmark = self.benchmark_capture_methods(timeout=timeout)
            if not benchmark.selected_method:
                raise AdbError(f"No usable ADB screenshot method found: {benchmark.summary()}")
        if self._selected_capture_method is None:
            raise AdbError("No usable ADB screenshot method found.")
        return self._selected_capture_method

    def _capture_frame_with_method(self, method: CaptureMethod, *, timeout: float) -> CapturedFrame:
        serial = self.selected_serial()
        started = perf_counter()
        if method == CaptureMethod.RAW:
            result = self._run(["-s", serial, "exec-out", "screencap"], text=False, timeout=timeout)
            payload = result.stdout
            frame = VisionFrame.from_raw_screencap(payload, source_format=method.value, started_at=started)
            return CapturedFrame(frame=frame, method=method, capture_ms=(perf_counter() - started) * 1000, raw_bytes=payload)
        if method == CaptureMethod.RAW_GZIP:
            result = self._run(
                ["-s", serial, "exec-out", "sh", "-c", "screencap | gzip -1"],
                text=False,
                timeout=timeout,
            )
            compressed = result.stdout
            try:
                payload = gzip.decompress(compressed)
            except OSError as exc:
                raise AdbError("ADB raw_gzip screencap did not return gzip data.") from exc
            frame = VisionFrame.from_raw_screencap(payload, source_format=method.value, started_at=started)
            return CapturedFrame(frame=frame, method=method, capture_ms=(perf_counter() - started) * 1000, raw_bytes=payload)
        if method == CaptureMethod.PNG:
            result = self._run(["-s", serial, "exec-out", "screencap", "-p"], text=False, timeout=timeout)
            image_bytes = _normalize_png_output(result.stdout)
            if not image_bytes.startswith(b"\x89PNG"):
                raise AdbError("ADB screencap did not return a PNG image.")
            frame = VisionFrame.from_bytes(image_bytes, source_format=method.value)
            return CapturedFrame(frame=frame, method=method, capture_ms=(perf_counter() - started) * 1000, raw_bytes=image_bytes)
        raise AdbError(f"Unsupported capture method: {method.value}")


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


def _normalize_png_output(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:120] or exc.__class__.__name__
