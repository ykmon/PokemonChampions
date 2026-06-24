import tempfile
import unittest
from pathlib import Path
import json

from champions_assistant.data_loader import DataRepository
from champions_assistant.config import AppConfig
from champions_assistant.models import Rect
from champions_assistant.preview_recognition import accepted_count, recognize_opponent_preview
from champions_assistant.templates import (
    PokemonTemplateMatcher,
    _cv2,
    _np,
    crop_image_bytes,
    default_opponent_preview_rois_1920,
    image_size_from_bytes,
)


def _has_cv2() -> bool:
    try:
        _cv2()
        _np()
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_cv2(), "opencv-python is not installed")
class TemplateTests(unittest.TestCase):
    def test_no_templates_returns_unknown(self):
        repository = DataRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = PokemonTemplateMatcher(repository, tmpdir)
            match = matcher.match(_solid_png())

        self.assertIsNone(match.species_id)
        self.assertEqual(match.confidence, 0.0)

    def test_saved_template_can_be_matched(self):
        repository = DataRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = PokemonTemplateMatcher(repository, tmpdir)
            image = _solid_png(color=(20, 80, 180))
            saved = matcher.save_template("swampert", image)
            match = matcher.match(image)

        self.assertTrue(saved.name.startswith("preview_"))
        self.assertEqual(match.species_id, "swampert")
        self.assertGreaterEqual(match.confidence, 0.99)

    def test_synthetic_template_can_match_unknown_metadata_species(self):
        repository = DataRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            species_dir = root / "no0003_pokemon"
            species_dir.mkdir()
            image = _solid_png(color=(20, 80, 180))
            (species_dir / "synthetic_redcard_001.png").write_bytes(image)
            (root / "template_metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "pokemon": {
                            "no0003_pokemon": {
                                "name_zh": "妙蛙花",
                                "pokedex_no": "0003",
                                "form_code": "00",
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            matcher = PokemonTemplateMatcher(repository, root)

            match = matcher.match(image)
            identity = matcher.match_identity(image)

        self.assertEqual(match.species_id, "no0003_pokemon")
        self.assertGreaterEqual(match.confidence, 0.99)
        self.assertEqual(identity.name, "妙蛙花")
        self.assertEqual(identity.source, "template")
        self.assertEqual(matcher.label_for_species("no0003_pokemon"), "妙蛙花")

    def test_default_rois_crop_six_non_empty_regions_from_1920_image(self):
        image = _blank_png(width=1920, height=1080)
        rois = default_opponent_preview_rois_1920()

        crops = [crop_image_bytes(image, rois[f"opponent_preview_{index}"]) for index in range(1, 7)]

        self.assertEqual(len(crops), 6)
        self.assertTrue(all(crop.startswith(b"\x89PNG") for crop in crops))

    def test_preview_recognition_returns_no_template_for_empty_repository(self):
        repository = DataRepository()
        config = AppConfig()
        image = _blank_png(width=1920, height=1080)
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = PokemonTemplateMatcher(repository, tmpdir)
            results = recognize_opponent_preview(config, repository, image, matcher=matcher)

        self.assertEqual(len(results), 6)
        self.assertEqual(accepted_count(results), 0)
        self.assertTrue(all(result.status == "no-template" for result in results))

    def test_preview_recognition_matches_cropped_slot_template(self):
        repository = DataRepository()
        config = AppConfig()
        rois = default_opponent_preview_rois_1920()
        slot_image = _solid_png(color=(20, 80, 180))
        image = _preview_screenshot_with_slot(3, slot_image, rois)
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = PokemonTemplateMatcher(repository, tmpdir)
            matcher.save_template("swampert", slot_image)
            results = recognize_opponent_preview(config, repository, image, matcher=matcher)

        slot = results[2]
        self.assertEqual(slot.slot_index, 3)
        self.assertEqual(slot.species_id, "swampert")
        self.assertEqual(slot.status, "accepted")
        self.assertGreaterEqual(slot.confidence, 0.99)
        self.assertGreaterEqual(accepted_count(results), 1)

    def test_preview_recognition_rejects_undecodable_image(self):
        repository = DataRepository()
        config = AppConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = PokemonTemplateMatcher(repository, tmpdir)
            with self.assertRaises(ValueError):
                recognize_opponent_preview(config, repository, b"not an image", matcher=matcher)


def _solid_png(width=150, height=115, color=(120, 30, 40)) -> bytes:
    cv2 = _cv2()
    np = _np()

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    cv2.line(image, (10, 10), (width - 10, height - 10), (255, 255, 255), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


def _blank_png(width=1920, height=1080) -> bytes:
    cv2 = _cv2()
    np = _np()

    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


def _preview_screenshot_with_slot(slot_index, slot_png, rois) -> bytes:
    cv2 = _cv2()
    np = _np()

    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    buffer = np.frombuffer(slot_png, dtype=np.uint8)
    slot = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    assert slot is not None
    rect = rois[f"opponent_preview_{slot_index}"]
    resized = cv2.resize(slot, (rect.width, rect.height), interpolation=cv2.INTER_AREA)
    image[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width] = resized
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


if __name__ == "__main__":
    unittest.main()
