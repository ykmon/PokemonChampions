from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .data_loader import DataRepository
from .models import Rect
from .templates import (
    AMBIGUITY_MARGIN_THRESHOLD,
    AUTO_ACCEPT_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MATCH_SIZE,
    SYNTHETIC_AUTO_ACCEPT_THRESHOLD,
    TEMPLATE_METADATA_NAME,
    TEMPLATE_ROOT,
    TemplateCandidate,
    TemplateMatch,
    _cv2,
    _metadata_label,
    _np,
)


@dataclass(frozen=True)
class VisionFrame:
    image: Any
    width: int
    height: int
    decode_ms: float
    source_format: str = "png"

    @classmethod
    def from_bytes(cls, image_bytes: bytes, *, source_format: str = "png") -> "VisionFrame":
        started = perf_counter()
        cv2 = _cv2()
        np = _np()
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bytes are not a decodable image.")
        height, width = image.shape[:2]
        return cls(image=image, width=width, height=height, decode_ms=(perf_counter() - started) * 1000, source_format=source_format)

    @classmethod
    def from_image(cls, image, *, decode_ms: float = 0.0, source_format: str = "bgr") -> "VisionFrame":
        height, width = image.shape[:2]
        return cls(image=image, width=width, height=height, decode_ms=decode_ms, source_format=source_format)

    @classmethod
    def from_raw_screencap(
        cls,
        data: bytes,
        *,
        source_format: str = "raw",
        started_at: float | None = None,
    ) -> "VisionFrame":
        started = started_at if started_at is not None else perf_counter()
        if len(data) < 12:
            raise ValueError("Raw screencap payload is too small.")
        width = int.from_bytes(data[0:4], "little", signed=False)
        height = int.from_bytes(data[4:8], "little", signed=False)
        if width <= 0 or height <= 0:
            raise ValueError("Raw screencap payload has an invalid size header.")
        pixel_size = 4 * width * height
        if len(data) < pixel_size + 8:
            raise ValueError("Raw screencap payload is shorter than its size header.")
        header_size = len(data) - pixel_size
        if header_size not in {12, 16}:
            raise ValueError(f"Raw screencap payload has an unsupported header size: {header_size}.")
        cv2 = _cv2()
        np = _np()
        rgba = np.frombuffer(data[header_size:], dtype=np.uint8).reshape((height, width, 4))
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return cls(image=image, width=width, height=height, decode_ms=(perf_counter() - started) * 1000, source_format=source_format)

    def roi(self, rect: Rect):
        safe = rect.clamp(self.width, self.height)
        if not safe.enabled:
            raise ValueError(f"ROI is empty after clamping: {rect}")
        x, y, w, h = safe.as_tuple()
        return self.image[y:y + h, x:x + w], safe

    def to_png_bytes(self) -> bytes:
        return _encode_png(self.image)


@dataclass(frozen=True)
class FastRecognitionTimings:
    decode_ms: float = 0.0
    crop_ms: float = 0.0
    feature_ms: float = 0.0
    coarse_match_ms: float = 0.0
    verify_ms: float = 0.0
    total_recognition_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "decode_ms": self.decode_ms,
            "crop_ms": self.crop_ms,
            "feature_ms": self.feature_ms,
            "coarse_match_ms": self.coarse_match_ms,
            "verify_ms": self.verify_ms,
            "total_recognition_ms": self.total_recognition_ms,
        }


@dataclass(frozen=True)
class FastPreviewSlot:
    slot_index: int
    roi_key: str
    rect: Rect
    match: TemplateMatch
    crop_image: Any
    crop_bytes: bytes
    elapsed_ms: float
    timings: FastRecognitionTimings
    failure_reason: str = ""


@dataclass(frozen=True)
class TemplateRecord:
    species_id: str
    path: Path
    feature: Any
    norm: float
    group: str


class TemplateBank:
    def __init__(
        self,
        repository: DataRepository,
        template_root: Path | str = TEMPLATE_ROOT,
        *,
        group: str = "preview",
    ) -> None:
        self.repository = repository
        self.template_root = Path(template_root)
        self.group = group
        self._records: tuple[TemplateRecord, ...] | None = None
        self._features = None
        self._norms = None
        self._metadata: dict[str, dict[str, Any]] | None = None

    @property
    def records(self) -> tuple[TemplateRecord, ...]:
        self._ensure_loaded()
        return self._records or ()

    @property
    def features(self):
        self._ensure_loaded()
        return self._features

    @property
    def norms(self):
        self._ensure_loaded()
        return self._norms

    def refresh(self) -> None:
        self._records = None
        self._features = None
        self._norms = None

    def label_for_species(self, species_id: str, language: str = "zh") -> str:
        if species_id in self.repository.pokemon_by_id:
            return self.repository.pokemon_label(species_id, language)
        metadata = self._metadata_for_species(species_id)
        if language == "zh" and metadata.get("name_zh"):
            return str(metadata["name_zh"])
        return _metadata_label(metadata, species_id)

    def _ensure_loaded(self) -> None:
        if self._records is not None:
            return
        cv2 = _cv2()
        np = _np()
        records: list[TemplateRecord] = []
        if self.template_root.exists():
            for species_dir in sorted(self.template_root.iterdir()):
                if not species_dir.is_dir():
                    continue
                for path in sorted(species_dir.glob("*.png")):
                    feature = _feature_from_bytes(path.read_bytes(), cv2=cv2, np=np)
                    records.append(
                        TemplateRecord(
                            species_id=species_dir.name,
                            path=path,
                            feature=feature,
                            norm=float(np.linalg.norm(feature)),
                            group=self.group,
                        )
                    )
        self._records = tuple(records)
        if not records:
            self._features = np.empty((0, MATCH_SIZE[0] * MATCH_SIZE[1] * 3), dtype="float32")
            self._norms = np.empty((0,), dtype="float32")
            return
        self._features = np.stack([record.feature for record in records]).astype("float32", copy=False)
        self._norms = np.array([record.norm for record in records], dtype="float32")

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
            raw = __import__("json").loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._metadata = {}
            return self._metadata
        pokemon = raw.get("pokemon", {})
        self._metadata = {str(key): value for key, value in pokemon.items() if isinstance(value, dict)} if isinstance(pokemon, dict) else {}
        return self._metadata


@dataclass(frozen=True)
class FastPreviewConfig:
    top1_threshold: float = AUTO_ACCEPT_THRESHOLD
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    margin_threshold: float = AMBIGUITY_MARGIN_THRESHOLD
    verify_top_k: int = 5
    min_template_votes: int = 1
    blank_std_threshold: float = 3.0


class FastPreviewRecognizer:
    def __init__(
        self,
        repository: DataRepository,
        *,
        bank: TemplateBank | None = None,
        config: FastPreviewConfig | None = None,
    ) -> None:
        self.repository = repository
        self.bank = bank or TemplateBank(repository)
        self.config = config or FastPreviewConfig()

    def recognize_slot(self, frame: VisionFrame, rect: Rect, *, slot_index: int, roi_key: str) -> FastPreviewSlot:
        return self.recognize_slots(frame, ((slot_index, roi_key, rect),))[0]

    def recognize_slots(self, frame: VisionFrame, slots: tuple[tuple[int, str, Rect], ...]) -> tuple[FastPreviewSlot, ...]:
        if not slots:
            return ()
        total_start = perf_counter()
        crop_start = perf_counter()
        cropped = []
        for slot_index, roi_key, rect in slots:
            crop_image, safe_rect = frame.roi(rect)
            cropped.append((slot_index, roi_key, safe_rect, crop_image))
        crop_ms = (perf_counter() - crop_start) * 1000
        feature_start = perf_counter()
        features = [_feature_from_image(crop_image) for _, _, _, crop_image in cropped]
        feature_ms = (perf_counter() - feature_start) * 1000
        match_start = perf_counter()
        matches = self._match_features(features, [crop_image for _, _, _, crop_image in cropped])
        coarse_ms = (perf_counter() - match_start) * 1000
        encode_start = perf_counter()
        crop_bytes_items = [_encode_png(crop_image) for _, _, _, crop_image in cropped]
        encode_ms = (perf_counter() - encode_start) * 1000
        total_ms = (perf_counter() - total_start) * 1000
        slot_count = len(cropped)
        per_slot_non_decode_ms = max(0.0, total_ms - encode_ms) / slot_count
        results: list[FastPreviewSlot] = []
        for index, ((slot_index, roi_key, safe_rect, crop_image), crop_bytes, (match, failure_reason)) in enumerate(
            zip(cropped, crop_bytes_items, matches)
        ):
            timings = FastRecognitionTimings(
                decode_ms=frame.decode_ms,
                crop_ms=crop_ms / slot_count,
                feature_ms=feature_ms / slot_count,
                coarse_match_ms=coarse_ms / slot_count,
                verify_ms=0.0,
                total_recognition_ms=frame.decode_ms + per_slot_non_decode_ms,
            )
            results.append(
                FastPreviewSlot(
                    slot_index=slot_index,
                    roi_key=roi_key,
                    rect=safe_rect,
                    match=match,
                    crop_image=crop_image,
                    crop_bytes=crop_bytes,
                    elapsed_ms=per_slot_non_decode_ms,
                    timings=timings,
                    failure_reason=failure_reason,
                )
            )
        return tuple(results)

    def _match_feature(self, feature, crop_image) -> tuple[TemplateMatch, str]:
        return self._match_features([feature], [crop_image])[0]

    def _match_features(self, features, crop_images) -> tuple[tuple[TemplateMatch, str], ...]:
        np = _np()
        records = self.bank.records
        if not records:
            return tuple((TemplateMatch(species_id=None, confidence=0.0), "no template candidates matched") for _ in features)

        feature_matrix = np.stack(features).astype("float32", copy=False)
        template_features = self.bank.features
        norms = self.bank.norms
        query_norms = np.linalg.norm(feature_matrix, axis=1)
        score_matrix = feature_matrix @ template_features.T / (query_norms[:, None] * norms[None, :] + 1e-9)
        results: list[tuple[TemplateMatch, str]] = []
        for index, scores in enumerate(score_matrix):
            if query_norms[index] <= 1e-9 or _is_blank(crop_images[index], self.config.blank_std_threshold):
                results.append((TemplateMatch(species_id=None, confidence=0.0), "roi appears blank or empty"))
                continue
            results.append(self._match_scores(scores))
        return tuple(results)

    def _match_scores(self, scores) -> tuple[TemplateMatch, str]:
        np = _np()
        records = self.bank.records
        top_count = min(max(2, self.config.verify_top_k), len(records))
        top_indices = np.argpartition(scores, -top_count)[-top_count:]
        ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        best_by_species: dict[str, tuple[float, TemplateRecord, int]] = {}
        for index in ranked_indices:
            record = records[int(index)]
            score = float(scores[int(index)])
            current = best_by_species.get(record.species_id)
            if current is None or score > current[0]:
                best_by_species[record.species_id] = (score, record, 1)
            else:
                best_by_species[record.species_id] = (current[0], current[1], current[2] + 1)
        ranked_species = sorted(best_by_species.values(), key=lambda item: item[0], reverse=True)
        best_score, best_record, best_votes = ranked_species[0]
        second_score = ranked_species[1][0] if len(ranked_species) > 1 else 0.0
        second_record = ranked_species[1][1] if len(ranked_species) > 1 else None
        candidates = tuple(
            TemplateCandidate(rank=rank, species_id=record.species_id, confidence=score, template_path=record.path)
            for rank, (score, record, _) in enumerate(ranked_species[:3], start=1)
        )
        threshold = SYNTHETIC_AUTO_ACCEPT_THRESHOLD if best_record.path.name.startswith("synthetic_redcard_") else self.config.top1_threshold
        failure_reason = ""
        if best_score < self.config.low_confidence_threshold:
            failure_reason = "confidence below low-confidence threshold"
        elif best_score < threshold:
            failure_reason = "confidence below auto-accept threshold"
        elif second_record is not None and best_score - second_score < self.config.margin_threshold:
            failure_reason = "top candidates are too close"
        elif best_votes < self.config.min_template_votes:
            failure_reason = "not enough matching template votes"

        match = TemplateMatch(
            species_id=best_record.species_id,
            confidence=best_score,
            template_path=best_record.path,
            second_confidence=second_score,
            second_species_id=second_record.species_id if second_record else None,
            second_template_path=second_record.path if second_record else None,
            candidates=candidates,
            auto_accept_threshold=threshold,
            low_confidence_threshold=self.config.low_confidence_threshold,
            ambiguity_margin_threshold=self.config.margin_threshold,
        )
        return match, failure_reason


def _feature_from_bytes(image_bytes: bytes, *, cv2=None, np=None):
    cv2 = cv2 or _cv2()
    np = np or _np()
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Template bytes are not a decodable image.")
    return _feature_from_image(image, cv2=cv2, np=np)


def _feature_from_image(image, *, cv2=None, np=None):
    cv2 = cv2 or _cv2()
    np = np or _np()
    resized = cv2.resize(image, MATCH_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    feature = cv2.merge((hsv[:, :, 0], hsv[:, :, 1], gray))
    feature = feature.astype("float32") / 255.0
    return feature.reshape(-1)


def _encode_png(image) -> bytes:
    cv2 = _cv2()
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode ROI crop as PNG.")
    return bytes(encoded)


def _is_blank(image, std_threshold: float) -> bool:
    np = _np()
    return float(np.std(image)) < std_threshold
