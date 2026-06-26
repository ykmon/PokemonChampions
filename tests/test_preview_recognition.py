import unittest
from unittest.mock import patch

from champions_assistant.config import AppConfig
from champions_assistant.data_loader import DataRepository
from champions_assistant.fast_preview import FastRecognitionTimings
from champions_assistant.models import Rect
from champions_assistant.preview_recognition import accepted_count, recognize_opponent_preview, recognize_opponent_preview_frame
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
        matcher = _FakeMatcher({3: TemplateMatch("swampert", 0.99, second_species_id="garchomp", second_confidence=0.73)})
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        slot = results[2]
        self.assertEqual(slot.slot_index, 3)
        self.assertEqual(slot.species_id, "swampert")
        self.assertEqual(slot.label, "Swampert")
        self.assertEqual(slot.status, "accepted")
        self.assertEqual(slot.confidence, 0.99)
        self.assertEqual(slot.second_species_id, "garchomp")
        self.assertEqual(slot.second_label, "Garchomp")
        self.assertEqual(slot.second_confidence, 0.73)
        self.assertEqual([candidate.species_id for candidate in slot.candidates], ["swampert", "garchomp"])
        self.assertEqual(slot.candidates[0].label, "Swampert")
        self.assertEqual(slot.thresholds["auto_accept"], 0.88)
        self.assertEqual(slot.failure_reason, "")
        self.assertGreaterEqual(slot.elapsed_ms, 0)
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
        self.assertEqual(results[0].failure_reason, "confidence below auto-accept threshold")
        self.assertEqual(results[1].status, "rejected")
        self.assertEqual(results[1].failure_reason, "confidence below low-confidence threshold")
        self.assertEqual(accepted_count(results), 0)

    def test_ambiguous_high_scores_are_not_accepted(self):
        matcher = _FakeMatcher({
            1: TemplateMatch("swampert", 0.96, second_species_id="garchomp", second_confidence=0.95),
        })
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(1920, 1080)), \
                patch("champions_assistant.preview_recognition.crop_image_bytes", side_effect=_fake_crop):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=matcher)

        self.assertEqual(results[0].status, "low-confidence")
        self.assertEqual(results[0].failure_reason, "top candidates are too close")
        self.assertEqual(accepted_count(results), 0)

    def test_undecodable_image_error_is_not_swallowed(self):
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", side_effect=ValueError("bad image")):
            with self.assertRaises(ValueError):
                recognize_opponent_preview(AppConfig(), DataRepository(), b"not an image", matcher=_FakeMatcher({}))

    def test_non_widescreen_image_is_rejected_before_matching(self):
        with patch("champions_assistant.preview_recognition.image_size_from_bytes", return_value=(2001, 856)):
            with self.assertRaisesRegex(ValueError, "not a 16:9 game screenshot"):
                recognize_opponent_preview(AppConfig(), DataRepository(), b"image", matcher=_FakeMatcher({}))

    def test_fast_preview_path_is_used_when_no_explicit_matcher(self):
        fake_frame = type("FakeFrame", (), {"width": 1920, "height": 1080, "decode_ms": 1.0})()
        fake_slots = tuple(
            type(
                "FakeSlot",
                (),
                {
                    "slot_index": index,
                    "roi_key": f"opponent_preview_{index}",
                    "match": TemplateMatch(None, 0.0),
                    "rect": Rect(1, 2, 3, 4),
                    "crop_bytes": b"crop",
                    "elapsed_ms": 1.2,
                    "timings": FastRecognitionTimings(decode_ms=1.0, total_recognition_ms=2.0),
                    "failure_reason": "no template candidates matched",
                },
            )()
            for index in range(1, 7)
        )
        with patch("champions_assistant.preview_recognition.VisionFrame.from_bytes", return_value=fake_frame) as decode, \
                patch("champions_assistant.preview_recognition.FastPreviewRecognizer.recognize_slots", return_value=fake_slots), \
                patch("champions_assistant.preview_recognition._cached_template_bank"):
            results = recognize_opponent_preview(AppConfig(), DataRepository(), b"image")

        decode.assert_called_once_with(b"image")
        self.assertEqual(len(results), 6)
        self.assertEqual(results[0].timings["total_recognition_ms"], 2.0)
        self.assertEqual([result.slot_index for result in results], [1, 2, 3, 4, 5, 6])
        self.assertEqual([result.roi_key for result in results], [f"opponent_preview_{index}" for index in range(1, 7)])

    def test_frame_preview_path_reuses_decoded_frame(self):
        fake_frame = type("FakeFrame", (), {"width": 1920, "height": 1080, "decode_ms": 0.2})()
        fake_slots = tuple(
            type(
                "FakeSlot",
                (),
                {
                    "slot_index": index,
                    "roi_key": f"opponent_preview_{index}",
                    "match": TemplateMatch(None, 0.0),
                    "rect": Rect(1, 2, 3, 4),
                    "crop_bytes": b"crop",
                    "elapsed_ms": 0.4,
                    "timings": FastRecognitionTimings(decode_ms=0.2, total_recognition_ms=0.6),
                    "failure_reason": "no template candidates matched",
                },
            )()
            for index in range(1, 7)
        )
        with patch("champions_assistant.preview_recognition.VisionFrame.from_bytes") as decode, \
                patch("champions_assistant.preview_recognition.FastPreviewRecognizer.recognize_slots", return_value=fake_slots), \
                patch("champions_assistant.preview_recognition._cached_template_bank"):
            results = recognize_opponent_preview_frame(AppConfig(), DataRepository(), fake_frame)

        decode.assert_not_called()
        self.assertEqual(len(results), 6)
        self.assertEqual(results[0].timings["decode_ms"], 0.2)


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
