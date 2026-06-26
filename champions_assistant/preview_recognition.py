from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import AppConfig
from .data_loader import DataRepository
from .fast_preview import FastPreviewConfig, FastPreviewRecognizer, TemplateBank, VisionFrame
from .models import Rect
from .templates import (
    PokemonTemplateMatcher,
    TemplateCandidate,
    TemplateMatch,
    crop_image_bytes,
    default_opponent_preview_rois_1920,
    image_size_from_bytes,
)
from .vision_config import build_template_matcher, load_vision_config

_FAST_BANK_CACHE: dict[tuple[str, str], TemplateBank] = {}


@dataclass(frozen=True)
class RecognitionCandidate:
    rank: int
    species_id: str
    label: str
    confidence: float
    template_path: Path | None = None


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
    second_species_id: str | None = None
    second_label: str = ""
    second_confidence: float = 0.0
    elapsed_ms: float = 0.0
    candidates: tuple[RecognitionCandidate, ...] = ()
    thresholds: dict[str, float] | None = None
    failure_reason: str = ""
    timings: dict[str, float] | None = None


def recognize_opponent_preview(
    config: AppConfig,
    repository: DataRepository,
    image_bytes: bytes,
    *,
    matcher: PokemonTemplateMatcher | None = None,
) -> list[PreviewRecognitionResult]:
    if matcher is None:
        vision_config = load_vision_config()
        if vision_config.enable_fast_preview:
            frame = VisionFrame.from_bytes(image_bytes)
            return recognize_opponent_preview_frame(config, repository, frame, vision_config=vision_config)
    matcher = matcher or build_template_matcher(repository)
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
        started = perf_counter()
        match = matcher.match(crop)
        elapsed_ms = (perf_counter() - started) * 1000
        label = "Unknown"
        status = "no-template"
        second_label = ""
        candidates = _recognition_candidates(match, matcher, config.language)
        if match.species_id:
            label = matcher.label_for_species(match.species_id, config.language)
            status = "accepted" if match.accepted else "low-confidence" if match.low_confidence else "rejected"
        if match.second_species_id:
            second_label = matcher.label_for_species(match.second_species_id, config.language)
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
                second_species_id=match.second_species_id,
                second_label=second_label,
                second_confidence=match.second_confidence,
                elapsed_ms=elapsed_ms,
                candidates=candidates,
                thresholds=_thresholds_from_match(match),
                failure_reason=_failure_reason(match, status),
            )
        )
    return results


def recognize_opponent_preview_frame(
    config: AppConfig,
    repository: DataRepository,
    frame: VisionFrame,
    *,
    vision_config=None,
) -> list[PreviewRecognitionResult]:
    vision_config = vision_config or load_vision_config()
    default_rois_1920 = default_opponent_preview_rois_1920()
    uses_default_rois = any(
        _uses_default_roi(config.rois.get(f"opponent_preview_{index}", Rect()), default_rois_1920[f"opponent_preview_{index}"])
        for index in range(1, 7)
    )
    if uses_default_rois:
        _validate_preview_aspect(frame.width, frame.height)
    scaled_default_rois = {
        key: _scale_rect(rect, frame.width, frame.height)
        for key, rect in default_rois_1920.items()
    }
    recognizer = FastPreviewRecognizer(
        repository,
        bank=_cached_template_bank(repository),
        config=FastPreviewConfig(
            top1_threshold=vision_config.auto_accept_threshold,
            low_confidence_threshold=vision_config.low_confidence_threshold,
            margin_threshold=vision_config.ambiguity_margin_threshold,
            verify_top_k=vision_config.verify_top_k,
            min_template_votes=vision_config.min_template_votes,
        ),
    )
    slot_inputs = []
    for index in range(1, 7):
        roi_key = f"opponent_preview_{index}"
        configured = config.rois.get(roi_key, Rect())
        rect = configured if configured.enabled and configured != default_rois_1920[roi_key] else scaled_default_rois[roi_key]
        slot_inputs.append((index, roi_key, rect))
    results: list[PreviewRecognitionResult] = []
    for slot in recognizer.recognize_slots(frame, tuple(slot_inputs)):
        match = slot.match
        label = "Unknown"
        status = "no-template"
        second_label = ""
        candidates = _recognition_candidates_from_bank(match, recognizer.bank, config.language)
        if match.species_id:
            label = recognizer.bank.label_for_species(match.species_id, config.language)
            status = "accepted" if match.accepted and not slot.failure_reason else "low-confidence" if match.low_confidence else "rejected"
        if match.second_species_id:
            second_label = recognizer.bank.label_for_species(match.second_species_id, config.language)
        results.append(
            PreviewRecognitionResult(
                slot_index=slot.slot_index,
                roi_key=slot.roi_key,
                label=label,
                species_id=match.species_id,
                confidence=match.confidence,
                status=status,
                template_path=match.template_path,
                crop_rect=slot.rect,
                crop_bytes=slot.crop_bytes,
                second_species_id=match.second_species_id,
                second_label=second_label,
                second_confidence=match.second_confidence,
                elapsed_ms=slot.elapsed_ms,
                candidates=candidates,
                thresholds=_thresholds_from_match(match),
                failure_reason=slot.failure_reason or _failure_reason(match, status),
                timings=slot.timings.as_dict(),
            )
        )
    return results


def _cached_template_bank(repository: DataRepository) -> TemplateBank:
    key = (str(getattr(repository, "data_dir", "")), "preview")
    bank = _FAST_BANK_CACHE.get(key)
    if bank is None:
        bank = TemplateBank(repository)
        _FAST_BANK_CACHE[key] = bank
    return bank


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


def _recognition_candidates(
    match: TemplateMatch,
    matcher: PokemonTemplateMatcher,
    language: str,
) -> tuple[RecognitionCandidate, ...]:
    raw_candidates = match.candidates
    if not raw_candidates and match.species_id:
        raw_candidates = (TemplateCandidate(1, match.species_id, match.confidence, match.template_path),)
        if match.second_species_id:
            raw_candidates = (
                *raw_candidates,
                TemplateCandidate(2, match.second_species_id, match.second_confidence, match.second_template_path),
            )
    candidates: list[RecognitionCandidate] = []
    for candidate in raw_candidates:
        candidates.append(
            RecognitionCandidate(
                rank=candidate.rank,
                species_id=candidate.species_id,
                label=matcher.label_for_species(candidate.species_id, language),
                confidence=candidate.confidence,
                template_path=candidate.template_path,
            )
        )
    return tuple(candidates)


def _recognition_candidates_from_bank(
    match: TemplateMatch,
    bank: TemplateBank,
    language: str,
) -> tuple[RecognitionCandidate, ...]:
    candidates: list[RecognitionCandidate] = []
    for candidate in match.candidates:
        candidates.append(
            RecognitionCandidate(
                rank=candidate.rank,
                species_id=candidate.species_id,
                label=bank.label_for_species(candidate.species_id, language),
                confidence=candidate.confidence,
                template_path=candidate.template_path,
            )
        )
    return tuple(candidates)


def _thresholds_from_match(match: TemplateMatch) -> dict[str, float]:
    return {
        "auto_accept": match.effective_auto_accept_threshold,
        "low_confidence": match.low_confidence_threshold,
        "ambiguity_margin": match.ambiguity_margin_threshold,
    }


def _failure_reason(match: TemplateMatch, status: str) -> str:
    if status == "accepted":
        return ""
    if match.species_id is None:
        return "no template candidates matched"
    if match.second_species_id and match.confidence - match.second_confidence < match.ambiguity_margin_threshold:
        return "top candidates are too close"
    if match.confidence < match.low_confidence_threshold:
        return "confidence below low-confidence threshold"
    if match.confidence < match.effective_auto_accept_threshold:
        return "confidence below auto-accept threshold"
    return status
