from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from .data_loader import DataRepository
from .paths import DEFAULT_VISION_CONFIG_PATH
from .templates import (
    AUTO_ACCEPT_THRESHOLD,
    AMBIGUITY_MARGIN_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    PokemonTemplateMatcher,
)


@dataclass(frozen=True)
class VisionConfig:
    use_enhanced_matching: bool = True
    enable_preprocessing: bool = True
    enable_verification: bool = True
    primary_algorithm: str = "TM_CCORR_NORMED"
    verification_algorithm: str = "TM_CCOEFF_NORMED"
    verification_threshold: float = 0.85
    feature_type: str = "adaptive"
    auto_accept_threshold: float = AUTO_ACCEPT_THRESHOLD
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    ambiguity_margin_threshold: float = AMBIGUITY_MARGIN_THRESHOLD
    enable_fast_preview: bool = True
    verify_top_k: int = 5
    min_template_votes: int = 1
    target_total_ms: float = 20.0


def load_vision_config(path: Path | str = DEFAULT_VISION_CONFIG_PATH) -> VisionConfig:
    config_path = Path(path)
    if not config_path.exists():
        return VisionConfig()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    recognition = raw.get("recognition", {})
    thresholds = recognition.get("thresholds", {})
    performance = recognition.get("performance", {})
    return VisionConfig(
        use_enhanced_matching=_bool(recognition.get("use_enhanced_matching"), True),
        enable_preprocessing=_bool(recognition.get("enable_preprocessing"), True),
        enable_verification=_bool(recognition.get("enable_verification"), True),
        primary_algorithm=str(recognition.get("primary_algorithm", "TM_CCORR_NORMED") or "TM_CCORR_NORMED"),
        verification_algorithm=str(recognition.get("verification_algorithm", "TM_CCOEFF_NORMED") or "TM_CCOEFF_NORMED"),
        verification_threshold=float(recognition.get("verification_threshold", 0.85) or 0.85),
        feature_type=str(recognition.get("feature_type", "adaptive") or "adaptive"),
        auto_accept_threshold=float(thresholds.get("auto_accept", AUTO_ACCEPT_THRESHOLD) or AUTO_ACCEPT_THRESHOLD),
        low_confidence_threshold=float(thresholds.get("low_confidence", LOW_CONFIDENCE_THRESHOLD) or LOW_CONFIDENCE_THRESHOLD),
        ambiguity_margin_threshold=float(
            thresholds.get("ambiguity_margin", AMBIGUITY_MARGIN_THRESHOLD) or AMBIGUITY_MARGIN_THRESHOLD
        ),
        enable_fast_preview=_bool(performance.get("enable_fast_preview"), True),
        verify_top_k=int(performance.get("verify_top_k", 5) or 5),
        min_template_votes=int(performance.get("min_template_votes", 1) or 1),
        target_total_ms=float(performance.get("target_total_ms", 20.0) or 20.0),
    )


def build_template_matcher(
    repository: DataRepository,
    *,
    template_root: Path | str | None = None,
    vision_config: VisionConfig | None = None,
) -> PokemonTemplateMatcher:
    config = vision_config or load_vision_config()
    kwargs = {
        "use_enhanced_matching": config.use_enhanced_matching,
        "enable_preprocessing": config.enable_preprocessing,
        "enable_verification": config.enable_verification,
        "primary_algorithm": config.primary_algorithm,
        "verification_algorithm": config.verification_algorithm,
        "verification_threshold": config.verification_threshold,
        "feature_type": config.feature_type,
        "auto_accept_threshold": config.auto_accept_threshold,
        "low_confidence_threshold": config.low_confidence_threshold,
        "ambiguity_margin_threshold": config.ambiguity_margin_threshold,
    }
    if template_root is None:
        return PokemonTemplateMatcher(repository, **kwargs)
    return PokemonTemplateMatcher(repository, template_root, **kwargs)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)
