from __future__ import annotations

from .models import Rect


class VisionDependencyError(RuntimeError):
    pass


def crop_png_with_cv2(image_bytes: bytes, rect: Rect) -> bytes:
    if not rect.enabled:
        return image_bytes
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise VisionDependencyError("Install the vision extra to crop screenshot ROIs.") from exc

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Input bytes are not a decodable image.")
    height, width = image.shape[:2]
    safe_rect = rect.clamp(width, height)
    if not safe_rect.enabled:
        return image_bytes
    x, y, w, h = safe_rect.as_tuple()
    crop = image[y:y + h, x:x + w]
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Failed to encode ROI crop as PNG.")
    return bytes(encoded)
