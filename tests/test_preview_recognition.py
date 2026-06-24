import unittest
from unittest.mock import patch

from champions_assistant.config import AppConfig
from champions_assistant.data_loader import DataRepository
from champions_assistant.preview_recognition import accepted_count, recognize_opponent_preview
from champions_assistant.templates import TemplateMatch


class PreviewRecognitionTests(unittest.TestCase):
    def test_no_templates_returns_six_no_template_results(self):
        matcher = _FakeMatcher({})
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        self.assertEqual(len(results), 6)
        self.assertEqual(accepted_count(results), 0)
        self.assertTrue(all(result.status == "no-template" for result in results))
        self.assertTrue(all(result.label == "Unknown" for result in results))

    def test_accepted_match_reports_label_and_confidence(self):
        matcher = _FakeMatcher({3: TemplateMatch("swampert", 0.99)})
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        slot = results[2]
        self.assertEqual(slot.slot_index, 3)
        self.assertEqual(slot.species_id, "swampert")
        self.assertEqual(slot.label, "Swampert")
        self.assertEqual(slot.status, "accepted")
        self.assertEqual(slot.confidence, 0.99)
        self.assertEqual(slot.crop_bytes, b"\x03")
        self.assertEqual((slot.crop_rect.x, slot.crop_rect.y, slot.crop_rect.width, slot.crop_rect.height), (1375, 404, 150, 115))
        self.assertEqual(accepted_count(results), 1)

    def test_low_confidence_and_rejected_statuses_follow_thresholds(self):
        matcher = _FakeMatcher({
            1: TemplateMatch("swampert", 0.60),
            2: TemplateMatch("garchomp", 0.20),
        })
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        self.assertEqual(results[0].status, "low-confidence")
        self.assertEqual(results[1].status, "rejected")
        self.assertEqual(accepted_count(results), 0)

    def test_ambiguous_high_scores_are_not_accepted(self):
        matcher = _FakeMatcher({
            1: TemplateMatch("swampert", 0.96, second_species_id="garchomp", second_confidence=0.95),
        })
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        self.assertEqual(results[0].status, "low-confidence")
        self.assertEqual(accepted_count(results), 0)

    def test_undecodable_image_error_is_not_swallowed(self):
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", side_effect=ValueError("bad image")):
            with self.assertRaises(ValueError):
                recognize_opponent_preview(AppConfig(), DataRepository(), b"not an image", matcher=_FakeMatcher({}))

    def test_non_widescreen_image_is_rejected_before_matching(self):
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(2001, 856)):
            with self.assertRaisesRegex(ValueError, "not a 16:9 game screenshot"):
                recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=_FakeMatcher({}))


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
    return bytes([_slot_index_from_rect(rect)])


def _slot_index_from_rect(rect):
    return ((rect.y - 144) // 130) + 1


if __name__ == "__main__":
    unittest.main()
