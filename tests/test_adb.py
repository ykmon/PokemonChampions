import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from champions_assistant.adb import AdbClient, DeviceInfo, parse_devices


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

        with tempfile.TemporaryDirectory() as tmpdir, patch("subprocess.run", fake_run):
            out = Path(tmpdir) / "shot.png"
            client = AdbClient("adb")
            data = client.capture_screenshot(out)
            written = out.read_bytes()

        self.assertEqual(data, b"\x89PNG\npayload")
        self.assertEqual(written, data)
        self.assertEqual(calls[-1][0], ["adb", "-s", "serial-1", "exec-out", "screencap", "-p"])


if __name__ == "__main__":
    unittest.main()
