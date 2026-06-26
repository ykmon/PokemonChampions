from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .config import AppConfig
from .data_loader import DataRepository
from .fast_preview import VisionFrame
from .models import Rect
from .preview_recognition import PreviewRecognitionResult, recognize_opponent_preview, recognize_opponent_preview_frame
from .state_machine import StateMachineResult, evaluate_readonly_state
from .templates import PokemonTemplateMatcher, image_size_from_bytes


@dataclass(frozen=True)
class PreviewDebugData:
    image_bytes: bytes
    image_size: tuple[int, int]
    results: tuple[PreviewRecognitionResult, ...]
    source: str = ""
    screen_name: str = "team_preview"
    state: StateMachineResult | None = None

    @property
    def low_quality_results(self) -> tuple[PreviewRecognitionResult, ...]:
        return tuple(result for result in self.results if result.status in {"low-confidence", "rejected", "no-template"})


def build_preview_debug_data(
    config: AppConfig,
    repository: DataRepository,
    image_bytes: bytes,
    *,
    source: str = "",
    matcher: PokemonTemplateMatcher | None = None,
) -> PreviewDebugData:
    image_size = image_size_from_bytes(image_bytes)
    results = tuple(recognize_opponent_preview(config, repository, image_bytes, matcher=matcher))
    state = evaluate_readonly_state(preview_results=results)
    return PreviewDebugData(
        image_bytes=image_bytes,
        image_size=image_size,
        results=results,
        source=source,
        screen_name=state.screen or "team_preview",
        state=state,
    )


def build_preview_debug_data_from_frame(
    config: AppConfig,
    repository: DataRepository,
    frame: VisionFrame,
    *,
    source: str = "",
) -> PreviewDebugData:
    image_bytes = frame.to_png_bytes()
    results = tuple(recognize_opponent_preview_frame(config, repository, frame))
    state = evaluate_readonly_state(preview_results=results)
    return PreviewDebugData(
        image_bytes=image_bytes,
        image_size=(frame.width, frame.height),
        results=results,
        source=source,
        screen_name=state.screen or "team_preview",
        state=state,
    )


def write_preview_debug_log(debug_data: PreviewDebugData, out_dir: Path | str) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "recognition.log.jsonl"
    payload = _debug_data_to_log(debug_data)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    return path


def export_low_confidence_samples(
    debug_data: PreviewDebugData,
    dataset_root: Path | str,
    *,
    batch_name: str | None = None,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch = batch_name or f"debug_preview_{stamp}"
    root = Path(dataset_root) / "pending" / batch
    root.mkdir(parents=True, exist_ok=True)
    source_name = "source.png"
    (root / source_name).write_bytes(debug_data.image_bytes)
    samples = []
    for result in debug_data.low_quality_results:
        sample_id = f"slot_{result.slot_index:02d}"
        crop_name = f"{sample_id}.png"
        (root / crop_name).write_bytes(result.crop_bytes)
        samples.append(
            {
                "sample_id": sample_id,
                "sample_type": "opponent_preview",
                "crop_path": crop_name,
                "predicted_species_id": result.species_id or "",
                "confidence": result.confidence,
                "second_species_id": result.second_species_id or "",
                "second_confidence": result.second_confidence,
                "status": result.status,
                "approved": False,
                "screen_name": debug_data.screen_name,
                "roi_key": result.roi_key,
                "candidates": [_candidate_to_dict(candidate) for candidate in result.candidates],
                "thresholds": result.thresholds or {},
                "source_image_path": source_name,
                "failure_reason": result.failure_reason,
                "roi": _rect_to_dict(result.crop_rect),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": debug_data.source,
                "source_image_path": source_name,
                "screen_name": debug_data.screen_name,
                "state": _state_to_dict(debug_data.state),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_review_html(root / "review.html", manifest.name, samples)
    return manifest


def _result_to_log(result: PreviewRecognitionResult) -> dict[str, object]:
    return {
        "slot_index": result.slot_index,
        "roi_key": result.roi_key,
        "label": result.label,
        "species_id": result.species_id,
        "confidence": result.confidence,
        "second_label": result.second_label,
        "second_species_id": result.second_species_id,
        "second_confidence": result.second_confidence,
        "status": result.status,
        "template_path": str(result.template_path) if result.template_path else "",
        "elapsed_ms": result.elapsed_ms,
        "performance": result.timings or {},
        "failure_reason": result.failure_reason,
        "thresholds": result.thresholds or {},
        "candidates": [_candidate_to_dict(candidate) for candidate in result.candidates],
        "roi": _rect_to_dict(result.crop_rect),
    }


def _rect_to_dict(rect: Rect) -> dict[str, int]:
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _debug_data_to_log(debug_data: PreviewDebugData) -> dict[str, object]:
    state = debug_data.state or evaluate_readonly_state(preview_results=debug_data.results)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": debug_data.source,
        "screen": debug_data.screen_name,
        "image_size": {"width": debug_data.image_size[0], "height": debug_data.image_size[1]},
        "state_machine": _state_to_dict(state),
        "performance": _aggregate_timings(debug_data),
        "results": [_result_to_log(result) for result in debug_data.results],
    }


def _state_to_dict(state: StateMachineResult | None) -> dict[str, object]:
    if state is None:
        return {}
    return {
        "state": state.state.value,
        "screen": state.screen,
        "should_refresh_recommendations": state.should_refresh_recommendations,
        "diagnostics": list(state.diagnostics),
        "message": state.message,
    }


def _aggregate_timings(debug_data: PreviewDebugData) -> dict[str, float]:
    result_timings = [result.timings or {} for result in debug_data.results]
    if not result_timings:
        return {}
    keys = ("decode_ms", "crop_ms", "feature_ms", "coarse_match_ms", "verify_ms", "total_recognition_ms")
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [float(timing.get(key, 0.0) or 0.0) for timing in result_timings]
        if key == "decode_ms":
            aggregate[key] = max(values)
        elif key == "total_recognition_ms":
            decode_ms = max(float(timing.get("decode_ms", 0.0) or 0.0) for timing in result_timings)
            aggregate[key] = decode_ms + sum(
                max(0.0, float(timing.get("total_recognition_ms", 0.0) or 0.0) - float(timing.get("decode_ms", 0.0) or 0.0))
                for timing in result_timings
            )
        else:
            aggregate[key] = sum(values)
    return aggregate


def _candidate_to_dict(candidate) -> dict[str, object]:
    return {
        "rank": candidate.rank,
        "species_id": candidate.species_id,
        "label": candidate.label,
        "confidence": candidate.confidence,
        "template_path": str(candidate.template_path) if candidate.template_path else "",
    }


def _write_review_html(path: Path, manifest_name: str, samples: list[dict[str, object]]) -> None:
    rows = []
    for sample in samples:
        candidates = ", ".join(
            f"{candidate.get('rank')}: {candidate.get('species_id')} {float(candidate.get('confidence', 0.0)):.3f}"
            for candidate in sample.get("candidates", [])
            if isinstance(candidate, dict)
        )
        rows.append(
            "<tr>"
            f"<td>{escape(str(sample.get('sample_id', '')))}</td>"
            f"<td><img src=\"{escape(str(sample.get('crop_path', '')))}\" alt=\"crop\" /></td>"
            f"<td>{escape(str(sample.get('predicted_species_id', '')))}</td>"
            f"<td>{float(sample.get('confidence', 0.0)):.3f}</td>"
            f"<td>{escape(candidates)}</td>"
            f"<td>{escape(str(sample.get('status', '')))}</td>"
            f"<td>{escape(str(sample.get('failure_reason', '')))}</td>"
            "<td><input type=\"text\" placeholder=\"actual_species_id\" /></td>"
            "<td><input type=\"checkbox\" /></td>"
            "</tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\" />"
        "<title>Pokemon Champions Review</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f8;color:#202124}"
        "table{border-collapse:collapse;width:100%;background:white}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}"
        "th{background:#f0f1f3} img{width:96px;height:96px;object-fit:contain}"
        "input[type=text]{width:160px}"
        "</style></head><body>"
        "<h1>Pending Recognition Review</h1>"
        f"<p>Manifest: {escape(manifest_name)}</p>"
        "<table><thead><tr>"
        "<th>Sample</th><th>Crop</th><th>Predicted</th><th>Confidence</th>"
        "<th>Candidates</th><th>Status</th><th>Failure</th><th>Actual species</th><th>Approve</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    path.write_text(html, encoding="utf-8")
