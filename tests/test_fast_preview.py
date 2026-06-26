import tempfile
import unittest
from pathlib import Path

from champions_assistant.data_loader import DataRepository
from champions_assistant.fast_preview import FastPreviewConfig, FastPreviewRecognizer, TemplateBank, VisionFrame
from champions_assistant.models import Rect
from champions_assistant.templates import _cv2, _np


def _has_cv2() -> bool:
    try:
        _cv2()
        _np()
    except Exception:
        return False
    return True


@unittest.skipUnless(_has_cv2(), "opencv-python is not installed")
class FastPreviewTests(unittest.TestCase):
    def test_vision_frame_decodes_once_and_crops_roi_view(self):
        frame = VisionFrame.from_bytes(_solid_png(width=320, height=180))

        crop, rect = frame.roi(Rect(10, 20, 30, 40))

        self.assertEqual((frame.width, frame.height), (320, 180))
        self.assertEqual(rect, Rect(10, 20, 30, 40))
        self.assertEqual(crop.shape[:2], (40, 30))
        self.assertGreaterEqual(frame.decode_ms, 0)

    def test_vision_frame_decodes_raw_screencap_rgba_to_bgr(self):
        np = _np()
        width, height = 2, 1
        rgba = np.array([[[10, 20, 30, 255], [40, 50, 60, 255]]], dtype=np.uint8)
        payload = (
            width.to_bytes(4, "little")
            + height.to_bytes(4, "little")
            + (1).to_bytes(4, "little")
            + rgba.tobytes()
        )

        frame = VisionFrame.from_raw_screencap(payload)

        self.assertEqual((frame.width, frame.height), (2, 1))
        self.assertEqual(frame.source_format, "raw")
        self.assertEqual(frame.image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(frame.image[0, 1].tolist(), [60, 50, 40])

    def test_template_bank_caches_and_refreshes_templates(self):
        repository = DataRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "swampert").mkdir()
            (root / "swampert" / "preview_001.png").write_bytes(_solid_png(color=(20, 80, 180)))
            bank = TemplateBank(repository, root)

            first_count = len(bank.records)
            (root / "garchomp").mkdir()
            (root / "garchomp" / "preview_001.png").write_bytes(_solid_png(color=(120, 10, 20)))
            cached_count = len(bank.records)
            bank.refresh()
            refreshed_count = len(bank.records)

        self.assertEqual(first_count, 1)
        self.assertEqual(cached_count, 1)
        self.assertEqual(refreshed_count, 2)

    def test_fast_recognizer_rejects_close_candidates(self):
        repository = DataRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for species in ("swampert", "garchomp"):
                (root / species).mkdir()
                (root / species / "preview_001.png").write_bytes(_solid_png(color=(20, 80, 180)))
            bank = TemplateBank(repository, root)
            recognizer = FastPreviewRecognizer(
                repository,
                bank=bank,
                config=FastPreviewConfig(top1_threshold=0.80, margin_threshold=0.20),
            )
            frame = VisionFrame.from_bytes(_solid_png(width=150, height=115, color=(20, 80, 180)))

            result = recognizer.recognize_slot(frame, Rect(0, 0, 150, 115), slot_index=1, roi_key="opponent_preview_1")

        self.assertEqual(result.failure_reason, "top candidates are too close")
        self.assertGreaterEqual(result.match.confidence, 0.99)
        self.assertEqual(len(result.match.candidates), 2)
        self.assertIn(result.match.species_id, {"swampert", "garchomp"})


def _solid_png(width=150, height=115, color=(120, 30, 40)) -> bytes:
    cv2 = _cv2()
    np = _np()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    cv2.line(image, (10, 10), (width - 10, height - 10), (255, 255, 255), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


if __name__ == "__main__":
    unittest.main()
