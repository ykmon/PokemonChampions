import tempfile
import unittest
from pathlib import Path

from champions_assistant.vision_config import load_vision_config


class VisionConfigTests(unittest.TestCase):
    def test_loads_fast_preview_performance_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vision.toml"
            path.write_text(
                "[recognition]\n"
                "use_enhanced_matching = false\n\n"
                "[recognition.thresholds]\n"
                "auto_accept = 0.93\n"
                "low_confidence = 0.61\n"
                "ambiguity_margin = 0.05\n\n"
                "[recognition.performance]\n"
                "enable_fast_preview = false\n"
                "verify_top_k = 7\n"
                "min_template_votes = 2\n"
                "target_total_ms = 12.5\n",
                encoding="utf-8",
            )

            config = load_vision_config(path)

        self.assertFalse(config.use_enhanced_matching)
        self.assertFalse(config.enable_fast_preview)
        self.assertEqual(config.verify_top_k, 7)
        self.assertEqual(config.min_template_votes, 2)
        self.assertEqual(config.target_total_ms, 12.5)
        self.assertEqual(config.auto_accept_threshold, 0.93)


if __name__ == "__main__":
    unittest.main()
