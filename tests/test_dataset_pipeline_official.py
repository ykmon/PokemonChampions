import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_pipeline():
    path = Path(r"C:\Users\zf18116\.codex\skills\pokemon-dataset-pipeline\scripts\pokemon_dataset_pipeline.py")
    spec = importlib.util.spec_from_file_location("pokemon_dataset_pipeline_skill", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_pillow() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_pillow(), "Pillow is not installed")
class OfficialSpritePipelineTests(unittest.TestCase):
    def test_import_synthesize_and_publish_official_sprite(self):
        pipeline = _load_pipeline()
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            web = root / "web"
            web.mkdir()
            sprite = Image.new("RGBA", (256, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(sprite)
            draw.rectangle((12, 12, 116, 116), fill=(40, 180, 80, 255))
            draw.ellipse((146, 16, 238, 108), fill=(210, 80, 60, 255))
            sprite.save(web / "sprite_poke_2.png")
            (web / "sprite_poke_2.css").write_text(
                """
.sprite-poke { background: url("./sprite_poke_2.png") no-repeat; }
.sprite-poke-ui_PokeIcon_02_0003_00_0 { background-size: 200% 100%; background-position: 0% 0%; }
.sprite-poke-ui_PokeIcon_02_0006_00_0 { background-size: 200% 100%; background-position: 100% 0%; }
""",
                encoding="utf-8",
            )
            (web / "pokemon.html").write_text(
                """
<html><head><link rel="stylesheet" href="sprite_poke_2.css"></head><body>
<div><div class="sprite-poke sprite-poke-ui_PokeIcon_02_0003_00_0"></div><div class="tooltip">No.0003 妙蛙花</div></div>
<div><div class="sprite-poke sprite-poke-ui_PokeIcon_02_0006_00_0"></div><div class="tooltip">No.0006 喷火龙</div></div>
</body></html>
""",
                encoding="utf-8",
            )

            dataset = root / "dataset"
            project_data = root / "project_data"
            project_data.mkdir()
            (project_data / "pokemon.json").write_text(
                json.dumps({"pokemon": [{"id": "venusaur", "name": "Venusaur", "name_zh": "妙蛙花"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (project_data / "aliases.json").write_text(
                json.dumps({"aliases": {"喷火龙": "charizard"}}, ensure_ascii=False),
                encoding="utf-8",
            )

            result = pipeline.import_official_icons(str(web / "pokemon.html"), dataset, "official_test", project_data)
            manifest_path = dataset / "pending" / "official_test" / "manifest.json"
            self.assertEqual(result, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertEqual(manifest["samples"][0]["predicted_name_zh"], "妙蛙花")
            self.assertEqual(manifest["samples"][0]["predicted_species_id"], "venusaur")
            self.assertEqual(manifest["samples"][1]["predicted_species_id"], "charizard")

            result = pipeline.synthesize_redcard_templates(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            synthetic = [sample for sample in manifest["samples"] if sample.get("sample_type") == "synthetic_redcard"]
            self.assertEqual(result, 0)
            self.assertEqual(len(synthetic), 8)
            self.assertTrue(all((manifest_path.parent / sample["crop_path"]).exists() for sample in synthetic))

            templates = root / "templates"
            result = pipeline.publish_templates(manifest_path, templates)
            metadata = json.loads((templates / "template_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertTrue((templates / synthetic[0]["predicted_species_id"] / "synthetic_redcard_001.png").exists())
            self.assertEqual(metadata["pokemon"][synthetic[0]["predicted_species_id"]]["name_zh"], "妙蛙花")


if __name__ == "__main__":
    unittest.main()
