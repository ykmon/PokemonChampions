import gzip
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from champions_assistant.adb import AdbClient, CapturedFrame, CaptureMethod, DeviceInfo, parse_devices


class AdbTests(unittest.TestCase):
    def test_parse_devices_keeps_status_and_details(self):
        output = """List of devices attached
emulator-5554	device product:sdk model:Android_SDK
127.0.0.1:5555	offline
"""
        devices = parse_devices(output)

        self.assertEqual(
            devices,
            [
                DeviceInfo("emulator-5554", "device", "product:sdk model:Android_SDK"),
                DeviceInfo("127.0.0.1:5555", "offline", ""),
            ],
        )
        self.assertTrue(devices[0].online)
        self.assertFalse(devices[1].online)

    def test_capture_screenshot_uses_exec_out_and_writes_png(self):
        calls = []

        def fake_run(command, check, capture_output, text, timeout):
            calls.append((command, text, timeout))
            if command[-2:] == ["devices", "-l"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="List of devices attached\nserial-1\tdevice\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(command, 0, stdout=b"\x89PNG\r\npayload", stderr=b"")

        with tempfile.TemporaryDirectory() as tmpdir, \
                patch("subprocess.run", fake_run), \
                patch("champions_assistant.adb.VisionFrame.from_bytes", return_value=_FakeFrame()):
            out = Path(tmpdir) / "shot.png"
            client = AdbClient("adb", emulator_profile="generic", capture_method="png")
            data = client.capture_screenshot(out)
            written = out.read_bytes()

        self.assertEqual(data, b"\x89PNG\npayload")
        self.assertEqual(written, data)
        self.assertEqual(calls[-1][0], ["adb", "-s", "serial-1", "exec-out", "screencap", "-p"])

    def test_capture_frame_raw_uses_unencoded_screencap(self):
        calls = []

        def fake_run(command, check, capture_output, text, timeout):
            calls.append((command, text, timeout))
            if command[-2:] == ["devices", "-l"]:
                return subprocess.CompletedProcess(command, 0, stdout="List of devices attached\nserial-1\tdevice\n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout=b"raw-payload", stderr=b"")

        with patch("subprocess.run", fake_run), \
                patch("champions_assistant.adb.VisionFrame.from_raw_screencap", return_value=_FakeFrame()) as decode:
            client = AdbClient("adb", emulator_profile="generic", capture_method="raw")
            captured = client.capture_frame()

        self.assertEqual(captured.method, CaptureMethod.RAW)
        self.assertEqual(captured.raw_bytes, b"raw-payload")
        self.assertEqual(calls[-1][0], ["adb", "-s", "serial-1", "exec-out", "screencap"])
        decode.assert_called_once()

    def test_capture_frame_raw_gzip_decompresses_before_decode(self):
        payload = b"raw-payload"
        compressed = gzip.compress(payload)

        def fake_run(command, check, capture_output, text, timeout):
            if command[-2:] == ["devices", "-l"]:
                return subprocess.CompletedProcess(command, 0, stdout="List of devices attached\nserial-1\tdevice\n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout=compressed, stderr=b"")

        with patch("subprocess.run", fake_run), \
                patch("champions_assistant.adb.VisionFrame.from_raw_screencap", return_value=_FakeFrame()) as decode:
            client = AdbClient("adb", emulator_profile="generic", capture_method="raw_gzip")
            captured = client.capture_frame()

        self.assertEqual(captured.method, CaptureMethod.RAW_GZIP)
        self.assertEqual(captured.raw_bytes, payload)
        self.assertEqual(decode.call_args.args[0], payload)

    def test_auto_capture_benchmark_selects_fastest_successful_method(self):
        client = AdbClient("adb", emulator_profile="generic")
        frame = _FakeFrame()

        def fake_capture(method, timeout):
            if method == CaptureMethod.RAW:
                raise ValueError("raw unsupported")
            if method == CaptureMethod.RAW_GZIP:
                return CapturedFrame(frame=frame, method=method, capture_ms=18.0, raw_bytes=b"raw")
            if method == CaptureMethod.PNG:
                return CapturedFrame(frame=frame, method=method, capture_ms=31.0, raw_bytes=b"\x89PNG\n")
            raise AssertionError(method)

        with patch.object(client, "_capture_frame_with_method", side_effect=fake_capture):
            benchmark = client.benchmark_capture_methods()
            captured = client.capture_frame()

        self.assertEqual(benchmark.selected_method, CaptureMethod.RAW_GZIP)
        self.assertEqual(captured.method, CaptureMethod.RAW_GZIP)
        self.assertIn("raw=fail", benchmark.summary())
        self.assertIn("raw_gzip=18.0ms", benchmark.summary())

    def test_ldplayer_auto_detect_selects_unique_emulator_serial(self):
        def fake_run(command, check, capture_output, text, timeout):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="List of devices attached\nemulator-5554\tdevice\n",
                stderr="",
            )

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb")
            discovery = client.discover_device()

        self.assertEqual(discovery.selected_serial, "emulator-5554")
        self.assertTrue(discovery.ok)
        self.assertIn("LDPlayer 9", discovery.message)

    def test_ldplayer_auto_detect_rejects_multiple_devices(self):
        def fake_run(command, check, capture_output, text, timeout):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="List of devices attached\nemulator-5554\tdevice\n127.0.0.1:5555\tdevice\n",
                stderr="",
            )

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb")
            with self.assertRaisesRegex(Exception, "Multiple LDPlayer 9 devices"):
                client.selected_serial()

    def test_ldplayer_auto_detect_tries_tcp_candidates(self):
        calls = []

        def fake_run(command, check, capture_output, text, timeout):
            calls.append(command)
            if command[-2:] == ["devices", "-l"]:
                devices_calls = sum(1 for call in calls if call[-2:] == ["devices", "-l"])
                stdout = "List of devices attached\n" if devices_calls == 1 else "List of devices attached\n127.0.0.1:5555\tdevice\n"
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
            if command[-2:] == ["connect", "127.0.0.1:5555"]:
                return subprocess.CompletedProcess(command, 0, stdout="connected to 127.0.0.1:5555\n", stderr="")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb", connect_timeout_ms=10)
            discovery = client.discover_device()

        self.assertEqual(discovery.selected_serial, "127.0.0.1:5555")
        self.assertIn(["adb", "connect", "127.0.0.1:5555"], calls)
        self.assertEqual(discovery.connected_tcp, ("127.0.0.1:5555",))

    def test_configured_serial_takes_precedence(self):
        calls = []

        def fake_run(command, check, capture_output, text, timeout):
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="List of devices attached\nserial-1\tdevice\n",
                stderr="",
            )

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb", "serial-1")
            self.assertEqual(client.selected_serial(), "serial-1")

        self.assertFalse(any(call[-2:] == ["connect", "127.0.0.1:5555"] for call in calls))

    def test_configured_missing_tcp_serial_attempts_connect(self):
        calls = []

        def fake_run(command, check, capture_output, text, timeout):
            calls.append(command)
            if command[-2:] == ["devices", "-l"]:
                devices_calls = sum(1 for call in calls if call[-2:] == ["devices", "-l"])
                stdout = "List of devices attached\n" if devices_calls == 1 else "List of devices attached\n127.0.0.1:5555\tdevice\n"
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="connected to 127.0.0.1:5555\n", stderr="")

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb", "127.0.0.1:5555", connect_timeout_ms=10)
            discovery = client.discover_device()

        self.assertEqual(discovery.selected_serial, "127.0.0.1:5555")
        self.assertEqual(discovery.attempted_tcp, ("127.0.0.1:5555",))

    def test_configured_missing_serial_errors(self):
        def fake_run(command, check, capture_output, text, timeout):
            return subprocess.CompletedProcess(command, 0, stdout="List of devices attached\n", stderr="")

        with patch("subprocess.run", fake_run):
            client = AdbClient("adb", "serial-1")
            with self.assertRaisesRegex(Exception, "Configured device_serial was not found"):
                client.discover_device()


class _FakeFrame:
    def to_png_bytes(self):
        return b"\x89PNG\nfake"


if __name__ == "__main__":
    unittest.main()
