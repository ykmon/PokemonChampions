import tempfile
import unittest
from pathlib import Path

from champions_assistant.models import Rect
from champions_assistant.screen_resources import load_resource_rois, load_screen_resources


class ScreenResourceTests(unittest.TestCase):
    def test_loads_screen_resource_rois(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            screens = root / "screens"
            screens.mkdir()
            (screens / "team_preview.toml").write_text(
                "[screen]\n"
                'name = "team_preview"\n'
                "base_width = 1920\n"
                "base_height = 1080\n"
                'recognizer = "template"\n\n'
                "[detection]\n"
                'kind = "opponent_preview_slots"\n\n'
                "[thresholds]\n"
                "auto_accept = 0.91\n"
                "low_confidence = 0.50\n\n"
                "[recognizers]\n"
                'opponent_preview_1 = "preview_template"\n\n'
                "[template_groups]\n"
                'opponent_preview_1 = "preview"\n\n'
                "[roi.opponent_preview_1]\n"
                "x = 10\n"
                "y = 20\n"
                "width = 30\n"
                "height = 40\n",
                encoding="utf-8",
            )

            resources = load_screen_resources(root)
            rois = load_resource_rois(root)

        self.assertIn("team_preview", resources)
        self.assertEqual(resources["team_preview"].recognizer, "template")
        self.assertEqual(resources["team_preview"].detection["kind"], "opponent_preview_slots")
        self.assertEqual(resources["team_preview"].thresholds["auto_accept"], 0.91)
        self.assertEqual(resources["team_preview"].recognizers["opponent_preview_1"], "preview_template")
        self.assertEqual(resources["team_preview"].template_groups["opponent_preview_1"], "preview")
        self.assertEqual(rois["opponent_preview_1"], Rect(10, 20, 30, 40))

    def test_default_resources_include_three_screen_protocols(self):
        resources = load_screen_resources()

        self.assertIn("team_preview", resources)
        self.assertIn("battle_active", resources)
        self.assertIn("result_screen", resources)
        self.assertEqual(resources["team_preview"].thresholds["top1_threshold"], 0.88)
        self.assertEqual(resources["team_preview"].thresholds["verify_top_k"], 5.0)
        self.assertEqual(resources["battle_active"].recognizers["opponent_active_1"], "active_name_ocr")
        self.assertEqual(resources["result_screen"].template_groups["result_banner"], "result_ui")


if __name__ == "__main__":
    unittest.main()
