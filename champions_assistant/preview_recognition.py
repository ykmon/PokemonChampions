from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .data_loader import DataRepository
from .models import Rect
from .templates import (
    PokemonTemplateMatcher,
    crop_image_bytes,
    default_opponent_preview_rois_1920,
    image_size_from_bytes,
)


@dataclass(frozen=True)
class PreviewRecognitionResult:
    slot_index: int
    roi_key: str
    label: str
    species_id: str | None
    confidence: float
    status: str
    template_path: Path | None
    crop_rect: Rect
    crop_bytes: bytes


def recognize_opponent_preview(
    config: AppConfig,
    repository: DataRepository,
    image_bytes: bytes,
    *,
    matcher: PokemonTemplateMatcher | None = None,
) -> list[PreviewRecognitionResult]:
    matcher = matcher or PokemonTemplateMatcher(repository)
    results: list[PreviewRecognitionResult] = []
    image_width, image_height = image_size_from_bytes(image_bytes)
    default_rois_1920 = default_opponent_preview_rois_1920()
    uses_default_rois = any(
        _uses_default_roi(config.rois.get(f"opponent_preview_{index}", Rect()), default_rois_1920[f"opponent_preview_{index}"])
        for index in range(1, 7)
    )
    if uses_default_rois:
        _validate_preview_aspect(image_width, image_height)
    scaled_default_rois = {
        key: _scale_rect(rect, image_width, image_height)
        for key, rect in default_rois_1920.items()
    }
    for index in range(1, 7):
        roi_key = f"opponent_preview_{index}"
        configured = config.rois.get(roi_key, Rect())
        rect = configured if configured.enabled and configured != default_rois_1920[roi_key] else scaled_default_rois[roi_key]
        crop = crop_image_bytes(image_bytes, rect)
        match = matcher.match(crop)
        label = "Unknown"
        status = "no-template"
        if match.species_id:
            label = matcher.label_for_species(match.species_id, config.language)
            status = "accepted" if match.accepted else "low-confidence" if match.low_confidence else "rejected"
        results.append(
            PreviewRecognitionResult(
                slot_index=index,
                roi_key=roi_key,
                label=label,
                species_id=match.species_id,
                confidence=match.confidence,
                status=status,
                template_path=match.template_path,
                crop_rect=rect,
                crop_bytes=crop,
            )
        )
    return results


def accepted_count(results: list[PreviewRecognitionResult]) -> int:
    return sum(1 for result in results if result.status == "accepted")


def _uses_default_roi(configured: Rect, default: Rect) -> bool:
    return not configured.enabled or configured == default


def _scale_rect(rect: Rect, image_width: int, image_height: int) -> Rect:
    x_scale = image_width / 1920
    y_scale = image_height / 1080
    return Rect(
        x=round(rect.x * x_scale),
        y=round(rect.y * y_scale),
        width=round(rect.width * x_scale),
        height=round(rect.height * y_scale),
    )


def _validate_preview_aspect(image_width: int, image_height: int) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Input image has an invalid size.")
    aspect = image_width / image_height
    expected = 16 / 9
    if abs(aspect - expected) > 0.08:
        raise ValueError(
            f"Input image is {image_width}x{image_height}, not a 16:9 game screenshot. "
            "Crop the emulator/game area first, then test recognition again."
        )
