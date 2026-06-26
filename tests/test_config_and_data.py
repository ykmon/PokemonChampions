import tempfile
import unittest
from pathlib import Path

from champions_assistant.config import AppConfig, load_config, save_config
from champions_assistant.data_loader import DataRepository
from champions_assistant.models import Rect


class ConfigAndDataTests(unittest.TestCase):
    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.toml"
            config = AppConfig()
            config.adb_path = "C:/Android/platform-tools/adb.exe"
            config.device_serial = "127.0.0.1:5555"
            config.emulator.profile = "ldplayer9"
            config.emulator.auto_detect = False
            config.emulator.connect_timeout_ms = 900
            config.emulator.capture_method = "raw"
            config.rois["self_name"] = Rect(10, 20, 130, 40)

            save_config(config, path)
            loaded = load_config(path)

        self.assertEqual(loaded.adb_path, config.adb_path)
        self.assertEqual(loaded.device_serial, config.device_serial)
        self.assertEqual(loaded.emulator.profile, "ldplayer9")
        self.assertFalse(loaded.emulator.auto_detect)
        self.assertEqual(loaded.emulator.connect_timeout_ms, 900)
        self.assertEqual(loaded.emulator.capture_method, "raw")
        self.assertEqual(loaded.rois["self_name"], Rect(10, 20, 130, 40))

    def test_legacy_config_gets_ldplayer_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.toml"
            path.write_text(
                'adb_path = "adb"\n'
                'device_serial = ""\n'
                'capture_interval_ms = 1200\n'
                'language = "zh"\n'
                'data_dir = "data"\n'
                'screenshots_dir = "screenshots"\n',
                encoding="utf-8",
            )

            loaded = load_config(path)

        self.assertEqual(loaded.emulator.profile, "ldplayer9")
        self.assertTrue(loaded.emulator.auto_detect)
        self.assertEqual(loaded.emulator.connect_timeout_ms, 1200)
        self.assertEqual(loaded.emulator.capture_method, "auto")

    def test_legacy_roi_migrates_to_active_slot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.toml"
            path.write_text(
                'adb_path = "adb"\n'
                'device_serial = ""\n'
                'capture_interval_ms = 1200\n'
                'language = "zh"\n'
                'data_dir = "data"\n'
                'screenshots_dir = "screenshots"\n\n'
                '[roi.self_name]\n'
                'x = 1\n'
                'y = 2\n'
                'width = 3\n'
                'height = 4\n',
                encoding="utf-8",
            )

            loaded = load_config(path)

        self.assertEqual(loaded.rois["player_active_1"], Rect(1, 2, 3, 4))
        self.assertEqual(loaded.rois["self_name"], Rect(1, 2, 3, 4))

    def test_user_roi_overrides_resource_defaults_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.toml"
            path.write_text(
                'adb_path = "adb"\n'
                'device_serial = ""\n'
                'capture_interval_ms = 1200\n'
                'language = "zh"\n'
                'data_dir = "data"\n'
                'screenshots_dir = "screenshots"\n\n'
                '[roi.opponent_preview_1]\n'
                'x = 11\n'
                'y = 22\n'
                'width = 33\n'
                'height = 44\n\n'
                '[roi.opponent_preview_2]\n'
                'x = 0\n'
                'y = 0\n'
                'width = 0\n'
                'height = 0\n',
                encoding="utf-8",
            )

            loaded = load_config(path)

        self.assertEqual(loaded.rois["opponent_preview_1"], Rect(11, 22, 33, 44))
        self.assertEqual(loaded.rois["opponent_preview_2"], Rect(1375, 274, 150, 115))

    def test_alias_resolution_supports_chinese_and_english(self):
        repository = DataRepository()

        self.assertEqual(repository.resolve_pokemon("皮卡丘").species_id, "pikachu")
        self.assertEqual(repository.resolve_pokemon("Flutter Mane").species_id, "flutter_mane")
        self.assertEqual(repository.resolve_pokemon("巨沼怪").species_id, "swampert")
        self.assertEqual(repository.resolve_pokemon("大嘴鸥").species_id, "pelipper")
        self.assertEqual(repository.resolve_pokemon("幽尾玄鱼").species_id, "basculegion")
        self.assertFalse(repository.resolve_pokemon("unknown").is_known)


if __name__ == "__main__":
    unittest.main()
