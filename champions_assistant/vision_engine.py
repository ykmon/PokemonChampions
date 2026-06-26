"""
Enhanced vision engine with multi-algorithm matching and preprocessing.
Inspired by MAA's robust recognition approach.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from .paths import PROJECT_ROOT
from .roi import VisionDependencyError


class MatchAlgorithm(str, Enum):
    """OpenCV template matching algorithms"""
    CCORR_NORMED = "TM_CCORR_NORMED"  # Fast, good for general cases
    CCOEFF_NORMED = "TM_CCOEFF_NORMED"  # Better for lighting variations
    SQDIFF_NORMED = "TM_SQDIFF_NORMED"  # Inverted score, good for verification


@dataclass(frozen=True)
class PreprocessConfig:
    """Image preprocessing configuration"""
    enable_clahe: bool = False  # Contrast Limited Adaptive Histogram Equalization
    clahe_clip_limit: float = 2.0
    enable_denoise: bool = False
    denoise_h: float = 10.0
    enable_sharpen: bool = False
    sharpen_amount: float = 1.0
    enable_gamma_correction: bool = False
    gamma: float = 1.0

    @classmethod
    def for_low_light(cls) -> PreprocessConfig:
        """Preset for low-light scenarios"""
        return cls(enable_clahe=True, enable_gamma_correction=True, gamma=1.2)

    @classmethod
    def for_blurry(cls) -> PreprocessConfig:
        """Preset for blurry images"""
        return cls(enable_sharpen=True, sharpen_amount=1.5, enable_denoise=True)

    @classmethod
    def for_noisy(cls) -> PreprocessConfig:
        """Preset for noisy screenshots"""
        return cls(enable_denoise=True, denoise_h=15.0)


@dataclass(frozen=True)
class MatchResult:
    """Template matching result with algorithm info"""
    confidence: float
    algorithm: MatchAlgorithm
    preprocessing: PreprocessConfig | None = None


class ImagePreprocessor:
    """Adaptive image preprocessing pipeline"""

    def __init__(self) -> None:
        self._cv2 = None
        self._np = None

    def preprocess(self, image, config: PreprocessConfig):
        """Apply preprocessing steps based on config"""
        cv2 = self._get_cv2()
        np = self._get_np()

        result = image.copy()

        # CLAHE for contrast enhancement
        if config.enable_clahe:
            if len(result.shape) == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)
                result = cv2.merge([l, a, b])
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8))
                result = clahe.apply(result)

        # Denoise
        if config.enable_denoise:
            if len(result.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(result, None, config.denoise_h, config.denoise_h, 7, 21)
            else:
                result = cv2.fastNlMeansDenoising(result, None, config.denoise_h, 7, 21)

        # Sharpen
        if config.enable_sharpen:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]], dtype=np.float32)
            kernel = kernel * config.sharpen_amount / 9
            kernel[1, 1] = kernel[1, 1] + (1 - config.sharpen_amount)
            result = cv2.filter2D(result, -1, kernel)

        # Gamma correction
        if config.enable_gamma_correction:
            inv_gamma = 1.0 / config.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            result = cv2.LUT(result, table)

        return result

    def analyze_image_quality(self, image) -> dict[str, float]:
        """Analyze image to suggest preprocessing"""
        cv2 = self._get_cv2()
        np = self._get_np()

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Brightness
        brightness = float(np.mean(gray))

        # Contrast
        contrast = float(np.std(gray))

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # Noise estimate (high-frequency content)
        noise = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise": noise,
        }

    def suggest_preprocessing(self, image) -> PreprocessConfig:
        """Auto-suggest preprocessing based on image quality"""
        quality = self.analyze_image_quality(image)

        config = PreprocessConfig()

        # Low brightness or contrast
        if quality["brightness"] < 80 or quality["contrast"] < 30:
            config = PreprocessConfig(enable_clahe=True, enable_gamma_correction=True, gamma=1.2)

        # Blurry
        elif quality["sharpness"] < 100:
            config = PreprocessConfig(enable_sharpen=True, sharpen_amount=1.5)

        # Noisy
        elif quality["noise"] > 15:
            config = PreprocessConfig(enable_denoise=True, denoise_h=12.0)

        return config

    def _get_cv2(self):
        if self._cv2 is None:
            self._cv2 = _vision_module("cv2")
        return self._cv2

    def _get_np(self):
        if self._np is None:
            self._np = _vision_module("numpy")
        return self._np


class MultiAlgorithmMatcher:
    """Template matcher using multiple algorithms for robustness"""

    def __init__(
        self,
        primary_algorithm: MatchAlgorithm = MatchAlgorithm.CCORR_NORMED,
        enable_verification: bool = True,
        verification_algorithm: MatchAlgorithm = MatchAlgorithm.CCOEFF_NORMED,
        verification_threshold: float = 0.85,
    ) -> None:
        self.primary_algorithm = primary_algorithm
        self.enable_verification = enable_verification
        self.verification_algorithm = verification_algorithm
        self.verification_threshold = verification_threshold
        self._cv2 = None
        self._np = None

    def match(
        self,
        query,
        template,
        *,
        preprocess_config: PreprocessConfig | None = None,
    ) -> MatchResult:
        """
        Match template with optional preprocessing and multi-algorithm verification.

        Args:
            query: Query image (preprocessed feature array)
            template: Template image (preprocessed feature array)
            preprocess_config: Optional preprocessing config used

        Returns:
            MatchResult with confidence and algorithm used
        """
        cv2 = self._get_cv2()

        # Primary match
        primary_score = self._match_with_algorithm(query, template, self.primary_algorithm)

        # Verification with different algorithm if primary confidence is high
        if self.enable_verification and primary_score >= self.verification_threshold:
            verification_score = self._match_with_algorithm(query, template, self.verification_algorithm)
            # Use average of both for final confidence
            final_score = (primary_score + verification_score) / 2.0
            return MatchResult(
                confidence=final_score,
                algorithm=self.primary_algorithm,
                preprocessing=preprocess_config,
            )

        return MatchResult(
            confidence=primary_score,
            algorithm=self.primary_algorithm,
            preprocessing=preprocess_config,
        )

    def match_multi_scale(
        self,
        query,
        template,
        scales: tuple[float, ...] = (0.9, 1.0, 1.1),
    ) -> MatchResult:
        """Match at multiple scales for robustness to size variations"""
        cv2 = self._get_cv2()

        best_score = 0.0
        best_algorithm = self.primary_algorithm

        for scale in scales:
            if abs(scale - 1.0) < 0.01:
                scaled_template = template
            else:
                h, w = template.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                scaled_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            try:
                result = self.match(query, scaled_template)
                if result.confidence > best_score:
                    best_score = result.confidence
                    best_algorithm = result.algorithm
            except cv2.error:
                # Template too large/small, skip this scale
                continue

        return MatchResult(confidence=best_score, algorithm=best_algorithm)

    def _match_with_algorithm(self, query, template, algorithm: MatchAlgorithm) -> float:
        """Perform template matching with specified algorithm"""
        cv2 = self._get_cv2()

        method = getattr(cv2, algorithm.value)
        result = cv2.matchTemplate(query, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # SQDIFF methods are inverted (lower is better)
        if algorithm == MatchAlgorithm.SQDIFF_NORMED:
            return 1.0 - float(min_val)

        return float(max_val)

    def _get_cv2(self):
        if self._cv2 is None:
            self._cv2 = _vision_module("cv2")
        return self._cv2

    def _get_np(self):
        if self._np is None:
            self._np = _vision_module("numpy")
        return self._np


class EnhancedFeatureExtractor:
    """Extract robust features from Pokemon preview images"""

    def __init__(
        self,
        feature_type: Literal["hsv_gray", "lab_gray", "hsv_only", "adaptive"] = "adaptive",
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.feature_type = feature_type
        self.target_size = target_size
        self._cv2 = None
        self._np = None
        self.preprocessor = ImagePreprocessor()

    def extract(
        self,
        image_bytes: bytes,
        *,
        preprocess_config: PreprocessConfig | None = None,
    ):
        """
        Extract features from image bytes.

        Args:
            image_bytes: PNG/JPEG image bytes
            preprocess_config: Optional preprocessing configuration

        Returns:
            Feature array (H, W, C) as float32 in [0, 1]
        """
        cv2 = self._get_cv2()
        np = self._get_np()

        # Decode
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bytes are not a decodable image.")

        # Resize first for performance
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Apply preprocessing if provided
        if preprocess_config:
            resized = self.preprocessor.preprocess(resized, preprocess_config)
        elif self.feature_type == "adaptive":
            # Auto-detect and preprocess
            suggested = self.preprocessor.suggest_preprocessing(resized)
            if suggested != PreprocessConfig():  # Not default
                resized = self.preprocessor.preprocess(resized, suggested)

        # Extract features based on type
        if self.feature_type == "hsv_gray":
            feature = self._extract_hsv_gray(resized, cv2)
        elif self.feature_type == "lab_gray":
            feature = self._extract_lab_gray(resized, cv2)
        elif self.feature_type == "hsv_only":
            feature = self._extract_hsv_only(resized, cv2)
        elif self.feature_type == "adaptive":
            # Use HSV+Gray by default, but could be smarter
            feature = self._extract_hsv_gray(resized, cv2)
        else:
            feature = self._extract_hsv_gray(resized, cv2)

        # Normalize to [0, 1]
        feature = feature.astype("float32") / 255.0
        return feature

    def _extract_hsv_gray(self, image, cv2):
        """Original: HSV (H, S) + Grayscale"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.merge((hsv[:, :, 0], hsv[:, :, 1], gray))

    def _extract_lab_gray(self, image, cv2):
        """Alternative: LAB (L, A, B) channels - better for color-similar Pokemon"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab

    def _extract_hsv_only(self, image, cv2):
        """HSV only - faster, good for colorful sprites"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _get_cv2(self):
        if self._cv2 is None:
            self._cv2 = _vision_module("cv2")
        return self._cv2

    def _get_np(self):
        if self._np is None:
            self._np = _vision_module("numpy")
        return self._np


def _vision_module(module_name: str, required_attr: str | None = None):
    """Import vision module with fallback to local .deps"""
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

    raise VisionDependencyError(
        f"Install the vision extra to use {module_name}: python -m pip install -e .[vision]"
    ) from first_error
