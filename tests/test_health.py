import unittest
from unittest.mock import patch

from champions_assistant.adb import DeviceDiscovery, DeviceInfo, EmulatorProfile
from champions_assistant.config import AppConfig
from champions_assistant.health import build_health_report


class HealthTests(unittest.TestCase):
    def test_health_report_includes_emulator_status(self):
        discovery = DeviceDiscovery(
            selected_serial="emulator-5554",
            devices=(DeviceInfo("emulator-5554", "device"),),
            attempted_tcp=(),
            connected_tcp=(),
            profile=EmulatorProfile.LDPLAYER9,
            message="selected LDPlayer 9 device: emulator-5554",
        )

        with patch("champions_assistant.health._check_data_repository", return_value=True), \
                patch("champions_assistant.health._check_vision", return_value=True), \
                patch("champions_assistant.health._check_ocr", return_value="none"), \
                patch("champions_assistant.health._count_templates", return_value=(1, 1)), \
                patch("champions_assistant.health.find_manifest_paths", return_value=()), \
                patch("champions_assistant.health.adb_client_from_config", return_value=_FakeClient(discovery)):
            report = build_health_report(AppConfig())

        names = [item.name for item in report.items]
        self.assertIn("emulator", names)
        emulator_item = next(item for item in report.items if item.name == "emulator")
        self.assertTrue(emulator_item.ok)
        self.assertIn("ldplayer9", emulator_item.detail)

    def test_health_report_lines_are_chinese(self):
        discovery = DeviceDiscovery(
            selected_serial=None,
            devices=(),
            attempted_tcp=(),
            connected_tcp=(),
            profile=EmulatorProfile.LDPLAYER9,
            message="",
        )

        with patch("champions_assistant.health._check_data_repository", return_value=True), \
                patch("champions_assistant.health._check_vision", return_value=True), \
                patch("champions_assistant.health._check_ocr", return_value="none"), \
                patch("champions_assistant.health._count_templates", return_value=(952, 236)), \
                patch("champions_assistant.health.find_manifest_paths", return_value=("manifest",)), \
                patch("champions_assistant.health.adb_client_from_config", return_value=_FakeClient(discovery, error="ADB executable not found: adb")):
            report = build_health_report(AppConfig())

        text = "\n".join(report.lines())

        self.assertIn("环境检查", text)
        self.assertIn("[正常] 数据", text)
        self.assertIn("OpenCV/Numpy 可用", text)
        self.assertIn("未检测到可用 OCR 引擎", text)
        self.assertIn("952 个模板文件，覆盖 236 个物种", text)
        self.assertIn("找不到 ADB 可执行文件: adb", text)
        self.assertIn("对手预览默认 ROI 生效", text)
        self.assertNotIn("Health check", text)
        self.assertNotIn("files across", text)


class _FakeClient:
    def __init__(self, discovery, error=""):
        self.discovery = discovery
        self.error = error

    def discover_device(self):
        if self.error:
            from champions_assistant.adb import AdbError

            raise AdbError(self.error)
        return self.discovery


if __name__ == "__main__":
    unittest.main()
