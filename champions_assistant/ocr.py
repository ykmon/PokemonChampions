from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .data_loader import DataRepository
from .models import (
    BattleFormat,
    BattleSide,
    BattleSnapshot,
    PokemonIdentity,
    FieldSlot,
    TeamSlot,
    merge_identity,
)
from .roi import VisionDependencyError, crop_png_with_cv2
from .templates import PokemonTemplateMatcher
from .vision_config import build_template_matcher


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float


class OptionalOcrEngine:
    def __init__(self) -> None:
        self.engine_name = "none"
        self._engine = None
        self._load_engine()

    @property
    def available(self) -> bool:
        return self._engine is not None

    def read_text(self, image_bytes: bytes) -> list[OcrText]:
        if self._engine is None:
            return []
        if self.engine_name == "rapidocr":
            return self._read_rapidocr(image_bytes)
        if self.engine_name == "paddleocr":
            return self._read_paddleocr(image_bytes)
        return []

    def _load_engine(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self.engine_name = "rapidocr"
            return
        except ImportError:
            pass

        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            self.engine_name = "paddleocr"
        except ImportError:
            self._engine = None
            self.engine_name = "none"

    def _decode_for_ocr(self, image_bytes: bytes):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise VisionDependencyError("Install the vision extra to run OCR.") from exc
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bytes are not a decodable image.")
        return image

    def _read_rapidocr(self, image_bytes: bytes) -> list[OcrText]:
        image = self._decode_for_ocr(image_bytes)
        result, _ = self._engine(image)
        texts: list[OcrText] = []
        for item in result or []:
            if len(item) >= 3:
                texts.append(OcrText(text=str(item[1]), confidence=float(item[2])))
        return texts

    def _read_paddleocr(self, image_bytes: bytes) -> list[OcrText]:
        image = self._decode_for_ocr(image_bytes)
        result = self._engine.ocr(image, cls=True)
        texts: list[OcrText] = []
        for line in result or []:
            for item in line or []:
                if len(item) >= 2:
                    text, confidence = item[1]
                    texts.append(OcrText(text=str(text), confidence=float(confidence)))
        return texts


class BattleRecognizer:
    def __init__(
        self,
        repository: DataRepository,
        config: AppConfig,
        engine: OptionalOcrEngine | None = None,
        template_matcher: PokemonTemplateMatcher | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self.engine = engine or OptionalOcrEngine()
        self.template_matcher = template_matcher or build_template_matcher(repository)

    def recognize(
        self,
        image_bytes: bytes,
        source_image: str = "",
        previous: BattleSnapshot | None = None,
        battle_format: BattleFormat | str | None = None,
    ) -> BattleSnapshot:
        fmt = BattleFormat.parse(battle_format or (previous.battle_format if previous else self.config.default_format))
        base = previous.with_format(fmt) if previous else BattleSnapshot.empty(fmt)

        player_team = self._recognize_team(image_bytes, "player", base.player_team)
        opponent_team = self._recognize_team(image_bytes, "opponent", base.opponent_team)
        player_active = self._recognize_active(image_bytes, "player", base.player_active)
        opponent_active = self._recognize_active(image_bytes, "opponent", base.opponent_active)
        turn_text = self._read_roi_text(image_bytes, "turn")
        return BattleSnapshot(
            battle_format=fmt,
            player_team=player_team,
            opponent_team=opponent_team,
            player_active=player_active,
            opponent_active=opponent_active,
            turn_text=turn_text,
            source_image=source_image,
        )

    def _recognize_team(
        self,
        image_bytes: bytes,
        side: BattleSide,
        previous_slots: tuple[TeamSlot, ...],
    ) -> tuple[TeamSlot, ...]:
        slots: list[TeamSlot] = []
        for slot in previous_slots:
            roi_key = f"{side}_preview_{slot.index}"
            if side == "opponent":
                recognized = self._recognize_template_roi(image_bytes, roi_key)
            else:
                recognized = self._recognize_roi(image_bytes, roi_key)
                if not recognized.is_known:
                    recognized = self._recognize_template_roi(image_bytes, roi_key)
            slots.append(
                TeamSlot(
                    side=slot.side,
                    index=slot.index,
                    pokemon=merge_identity(slot.pokemon, recognized),
                    selected=slot.selected,
                    locked=slot.locked,
                )
            )
        return tuple(slots)

    def _recognize_active(
        self,
        image_bytes: bytes,
        side: BattleSide,
        previous_slots: tuple[FieldSlot, ...],
    ) -> tuple[FieldSlot, ...]:
        slots: list[FieldSlot] = []
        for slot in previous_slots:
            roi_key = f"{side}_active_{slot.index}"
            hp_key = f"{side}_active_{slot.index}_hp"
            recognized = self._recognize_roi(image_bytes, roi_key)
            hp_text = self._read_roi_text(image_bytes, hp_key) or slot.hp_text
            slots.append(
                FieldSlot(
                    side=slot.side,
                    index=slot.index,
                    pokemon=merge_identity(slot.pokemon, recognized),
                    hp_text=hp_text,
                    status_text=slot.status_text,
                    team_slot_index=slot.team_slot_index,
                    locked=slot.locked,
                )
            )
        return tuple(slots)

    def _recognize_roi(self, image_bytes: bytes, roi_key: str) -> PokemonIdentity:
        texts = self._read_roi(image_bytes, roi_key)
        best: PokemonIdentity | None = None
        for ocr_text in texts:
            identity = self.repository.resolve_pokemon(ocr_text.text, confidence=ocr_text.confidence, source="ocr")
            if identity.is_known and (best is None or identity.confidence > best.confidence):
                best = identity
        return best or PokemonIdentity(source="ocr")

    def _recognize_template_roi(self, image_bytes: bytes, roi_key: str) -> PokemonIdentity:
        rect = self.config.rois.get(roi_key)
        if not rect or not rect.enabled:
            return PokemonIdentity(source="template")
        try:
            crop = crop_png_with_cv2(image_bytes, rect)
            return self.template_matcher.match_identity(crop)
        except (VisionDependencyError, ValueError):
            return PokemonIdentity(source="template")

    def _read_roi_text(self, image_bytes: bytes, roi_key: str) -> str:
        texts = self._read_roi(image_bytes, roi_key)
        return " ".join(text.text for text in texts).strip()

    def _read_roi(self, image_bytes: bytes, roi_key: str) -> list[OcrText]:
        rect = self.config.rois.get(roi_key)
        target = image_bytes
        if rect and rect.enabled:
            target = crop_png_with_cv2(image_bytes, rect)
        return self.engine.read_text(target)
