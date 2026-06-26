import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from champions_assistant.config import AppConfig
from champions_assistant.data_loader import DataRepository
from champions_assistant.debug_tools import (
    build_preview_debug_data,
    export_low_confidence_samples,
    write_preview_debug_log,
)
from champions_assistant.templates import TemplateMatch


class DebugToolsTests(unittest.TestCase):
    def test_debug_data_log_and_export(self):
        matcher = _FakeMatcher({
            1: TemplateMatch("swampert", 0.60),
            2: TemplateMatch("garchomp", 0.99),
        })
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.debug_tools.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            debug = build_preview_debug_data(AppConfig(), DataRepository(), b"image", source="unit", matcher=matcher)

        self.assertEqual(debug.image_size, (1920, 1080))
        self.assertEqual(len(debug.results), 6)
        self.assertEqual(debug.results[0].status, "low-confidence")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_preview_debug_log(debug, Path(tmpdir) / "logs")
            manifest_path = export_low_confidence_samples(debug, Path(tmpdir) / "dataset", batch_name="batch")
            log_lines = log_path.read_text(encoding="utf-8").splitlines()
            log = json.loads(log_lines[-1])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            review_path = manifest_path.with_name("review.html")
            review_exists = review_path.exists()
            source_exists = manifest_path.with_name("source.png").exists()

        self.assertEqual(log["source"], "unit")
        self.assertEqual(log["screen"], "team_preview")
        self.assertEqual(log["state_machine"]["screen"], "team_preview")
        self.assertIn("performance", log)
        self.assertEqual(len(log["results"]), 6)
        self.assertIn("candidates", log["results"][0])
        self.assertIn("thresholds", log["results"][0])
        self.assertIn("performance", log["results"][0])
        self.assertTrue(manifest["samples"])
        self.assertEqual(manifest["screen_name"], "team_preview")
        self.assertEqual(manifest["source_image_path"], "source.png")
        self.assertEqual(manifest["samples"][0]["sample_type"], "opponent_preview")
        self.assertEqual(manifest["samples"][0]["screen_name"], "team_preview")
        self.assertEqual(manifest["samples"][0]["roi_key"], "opponent_preview_1")
        self.assertIn("candidates", manifest["samples"][0])
        self.assertIn("thresholds", manifest["samples"][0])
        self.assertIn("failure_reason", manifest["samples"][0])
        self.assertTrue(review_exists)
        self.assertTrue(source_exists)


class _FakeMatcher:
    def __init__(self, matches):
        self.matches = matches

    def match(self, crop):
        return self.matches.get(crop[0], TemplateMatch(None, 0.0))

    def label_for_species(self, species_id, language="zh"):
        return {
            "swampert": "Swampert",
            "garchomp": "Garchomp",
        }.get(species_id, species_id)


def _fake_crop(image_bytes, rect):
    return bytes([((rect.y - 144) // 130) + 1])


if __name__ == "__main__":
    unittest.main()
