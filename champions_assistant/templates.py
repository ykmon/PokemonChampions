from __future__ import annotations

import json
import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .data_loader import DataRepository
from .models import PokemonIdentity, Rect
from .paths import PROJECT_ROOT
from .roi import VisionDependencyError


TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "pokemon_templates"
TEMPLATE_METADATA_NAME = "template_metadata.json"
MATCH_SIZE = (128, 128)
AUTO_ACCEPT_THRESHOLD = 0.88
SYNTHETIC_AUTO_ACCEPT_THRESHOLD = 0.965
LOW_CONFIDENCE_THRESHOLD = 0.55
AMBIGUITY_MARGIN_THRESHOLD = 0.025


@dataclass(frozen=True)
class TemplateMatch:
    species_id: str | None
    confidence: float
    template_path: Path | None = None
    second_confidence: float = 0.0
    second_species_id: str | None = None

    @property
    def accepted(self) -> bool:
        if self.species_id is None:
            return False
        threshold = _threshold_for_template(self.template_path)
        if self.confidence < threshold:
            return False
        if self.second_species_id and self.confidence - self.second_confidence < AMBIGUITY_MARGIN_THRESHOLD:
            return False
        return True

    @property
    def low_confidence(self) -> bool:
        return self.species_id is not None and LOW_CONFIDENCE_THRESHOLD <= self.confidence and not self.accepted


def default_opponent_preview_rois_1920() -> dict[str, Rect]:
    return {
        f"opponent_preview_{index}": Rect(x=1375, y=144 + (index - 1) * 130, width=150, height=115)
        for index in range(1, 7)
    }


class PokemonTemplateMatcher:
    def __init__(self, repository: DataRepository, template_root: Path | str = TEMPLATE_ROOT) -> None:
        self.repository = repository
        self.template_root = Path(template_root)
        self._templates: list[tuple[str, Path, object]] | None = None
        self._metadata: dict[str, dict[str, Any]] | None = None

    def match_identity(self, image_bytes: bytes) -> PokemonIdentity:
        match = self.match(image_bytes)
        if not match.accepted or not match.species_id:
            return PokemonIdentity(source="template", confidence=match.confidence)
        if match.species_id not in self.repository.pokemon_by_id:
            metadata = self._metadata_for_species(match.species_id)
            return PokemonIdentity(
                name=_metadata_label(metadata, match.species_id),
                species_id=match.species_id,
                confidence=match.confidence,
                source="template",
            )
        return self.repository.identity_for_id(match.species_id, confidence=match.confidence, source="template")

    def label_for_species(self, species_id: str, language: str = "zh") -> str:
        if species_id in self.repository.pokemon_by_id:
            return self.repository.pokemon_label(species_id, language)
        metadata = self._metadata_for_species(species_id)
        if language == "zh" and metadata.get("name_zh"):
            return str(metadata["name_zh"])
        return _metadata_label(metadata, species_id)

    def match(self, image_bytes: bytes) -> TemplateMatch:
        templates = self._load_templates()
        if not templates:
            return TemplateMatch(species_id=None, confidence=0.0)
        query = _prepare_image(image_bytes)
        best_by_species: dict[str, TemplateMatch] = {}
        cv2 = _cv2()
        for species_id, path, template in templates:
            result = cv2.matchTemplate(query, template, cv2.TM_CCORR_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            species_best = best_by_species.get(species_id)
            if species_best is None or float(score) > species_best.confidence:
                best_by_species[species_id] = TemplateMatch(species_id=species_id, confidence=float(score), template_path=path)
        if not best_by_species:
            return TemplateMatch(species_id=None, confidence=0.0)
        ranked = sorted(best_by_species.values(), key=lambda match: match.confidence, reverse=True)
        best = ranked[0]
        if len(ranked) == 1:
            return best
        runner_up = ranked[1]
        return replace(best, second_confidence=runner_up.confidence, second_species_id=runner_up.species_id)

    def save_template(self, species_id: str, image_bytes: bytes) -> Path:
        if species_id not in self.repository.pokemon_by_id and species_id not in self._load_metadata():
            raise KeyError(f"Unknown species_id: {species_id}")
        species_dir = self.template_root / species_id
        species_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(species_dir.glob("preview_*.png"))
        next_index = len(existing) + 1
        out_path = species_dir / f"preview_{next_index:03d}.png"
        out_path.write_bytes(image_bytes)
        self._templates = None
        return out_path

    def _load_templates(self) -> list[tuple[str, Path, object]]:
        if self._templates is not None:
            return self._templates
        templates: list[tuple[str, Path, object]] = []
        if not self.template_root.exists():
            self._templates = []
            return self._templates
        for species_dir in sorted(self.template_root.iterdir()):
            if not species_dir.is_dir():
                continue
            for path in sorted(species_dir.glob("*.png")):
                templates.append((species_dir.name, path, _prepare_image(path.read_bytes())))
        self._templates = templates
        return self._templates

    def _metadata_for_species(self, species_id: str) -> dict[str, Any]:
        return self._load_metadata().get(species_id, {})

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if self._metadata is not None:
            return self._metadata
        path = self.template_root / TEMPLATE_METADATA_NAME
        if not path.exists():
            self._metadata = {}
            return self._metadata
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._metadata = {}
            return self._metadata
        pokemon = raw.get("pokemon", {})
        if isinstance(pokemon, dict):
            self._metadata = {str(key): value for key, value in pokemon.items() if isinstance(value, dict)}
        else:
            self._metadata = {}
        return self._metadata


def crop_image_bytes(image_bytes: bytes, rect: Rect) -> bytes:
    cv2 = _cv2()
    np = _np()
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Input bytes are not a decodable image.")
    height, width = image.shape[:2]
    safe_rect = rect.clamp(width, height)
    if not safe_rect.enabled:
        raise ValueError(f"ROI is empty after clamping: {rect}")
    x, y, w, h = safe_rect.as_tuple()
    crop = image[y:y + h, x:x + w]
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Failed to encode template crop.")
    return bytes(encoded)


def image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    cv2 = _cv2()
    np = _np()
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Input bytes are not a decodable image.")
    height, width = image.shape[:2]
    return width, height


def _prepare_image(image_bytes: bytes):
    cv2 = _cv2()
    np = _np()
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Template bytes are not a decodable image.")
    resized = cv2.resize(image, MATCH_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    feature = cv2.merge((hsv[:, :, 0], hsv[:, :, 1], gray))
    feature = feature.astype("float32") / 255.0
    return feature


def _metadata_label(metadata: dict[str, Any], fallback: str) -> str:
    return str(metadata.get("name_en") or metadata.get("name_zh") or fallback)


def _cv2():
    return _vision_module("cv2", required_attr="imdecode")


def _np():
    return _vision_module("numpy", required_attr="frombuffer")


def _threshold_for_template(path: Path | None) -> float:
    if path and path.name.startswith("synthetic_redcard_"):
        return SYNTHETIC_AUTO_ACCEPT_THRESHOLD
    return AUTO_ACCEPT_THRESHOLD


def _vision_module(module_name: str, required_attr: str | None = None):
    first_error: Exception | None = None
    try:
        module = importlib.import_module(module_name)
        if required_attr is None or hasattr(module, required_attr):
            return module
        first_error = ImportError(f"{module_name} is missing {required_attr}")
        sys.modules.pop(module_name, None)
    except ImportError as exc:
        first_error = exc

    local_vision = PROJECT_ROOT / ".deps" / "vision"
    if local_vision.exists():
        local_vision_text = str(local_vision)
        if local_vision_text not in sys.path:
            sys.path.insert(0, local_vision_text)
        try:
            module = importlib.import_module(module_name)
            if required_attr is None or hasattr(module, required_attr):
                return module
            first_error = ImportError(f"{module_name} is missing {required_attr}")
        except ImportError as exc:
            first_error = exc

    raise VisionDependencyError("Install the vision extra to use template matching: python -m pip install -e .[vision]") from first_error
