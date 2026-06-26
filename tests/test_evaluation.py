import json
import tempfile
import unittest
from pathlib import Path

from champions_assistant.evaluation import (
    DATASET_STAGES,
    ensure_dataset_layout,
    evaluate_template_manifests,
    find_manifest_paths,
    load_evaluation_samples,
)
from champions_assistant.templates import TemplateMatch


class EvaluationTests(unittest.TestCase):
    def test_dataset_layout_creates_stage_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            created = ensure_dataset_layout(root)

            self.assertEqual({path.name for path in created}, set(DATASET_STAGES))
            self.assertTrue(all((root / stage).is_dir() for stage in DATASET_STAGES))

    def test_manifest_loading_skips_unapproved_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir), [
                {"sample_id": "ok", "crop_path": "ok.png", "predicted_species_id": "pikachu", "approved": True},
                {"sample_id": "draft", "crop_path": "draft.png", "predicted_species_id": "gengar", "approved": False},
            ])

            samples = load_evaluation_samples([manifest])
            all_samples = load_evaluation_samples([manifest], include_unapproved=True)

            self.assertEqual([sample.sample_id for sample in samples], ["ok"])
            self.assertEqual([sample.sample_id for sample in all_samples], ["ok", "draft"])

    def test_evaluation_reports_accuracy_confusions_and_species_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = _write_manifest(root, [
                {"sample_id": "one", "crop_path": "one.png", "predicted_species_id": "pikachu", "approved": True},
                {"sample_id": "two", "crop_path": "two.png", "predicted_species_id": "gengar", "approved": True},
            ])
            (root / "one.png").write_bytes(b"one")
            (root / "two.png").write_bytes(b"two")
            matcher = _FakeMatcher({
                b"one": TemplateMatch("pikachu", 0.99),
                b"two": TemplateMatch("pikachu", 0.99),
            })

            report = evaluate_template_manifests([manifest], matcher)

            self.assertEqual(report.sample_count, 2)
            self.assertEqual(report.accepted_count, 2)
            self.assertEqual(report.correct_count, 1)
            self.assertEqual(report.accepted_precision, 0.5)
            self.assertGreaterEqual(report.p95_ms, 0.0)
            self.assertEqual(report.confusions[("gengar", "pikachu")], 1)
            self.assertEqual(report.species_metrics["pikachu"].accepted_correct, 1)
            self.assertEqual(report.species_metrics["gengar"].accepted_correct, 0)

    def test_find_manifest_paths_scans_lifecycle_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_dataset_layout(root)
            manifest = _write_manifest(root / "approved" / "batch", [])

            paths = find_manifest_paths(root, "approved")

            self.assertEqual(paths, (manifest,))


class _FakeMatcher:
    def __init__(self, matches):
        self.matches = matches

    def match(self, image_bytes):
        return self.matches.get(image_bytes, TemplateMatch(None, 0.0))


def _write_manifest(root, samples):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    unittest.main()
