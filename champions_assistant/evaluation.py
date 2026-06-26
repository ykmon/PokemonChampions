from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import Iterable, Sequence

from .templates import PokemonTemplateMatcher, TemplateMatch


DATASET_STAGES = ("pending", "reviewed", "approved", "rejected")


@dataclass(frozen=True)
class EvaluationSample:
    manifest_path: Path
    sample_id: str
    crop_path: Path
    expected_species_id: str
    approved: bool | None = None
    sample_type: str = ""
    notes: str = ""


@dataclass(frozen=True)
class SampleEvaluation:
    sample: EvaluationSample
    predicted_species_id: str | None
    confidence: float
    accepted: bool
    status: str
    correct: bool
    top1_correct: bool
    second_species_id: str | None = None
    second_confidence: float = 0.0
    error: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class SpeciesMetrics:
    total: int
    accepted_correct: int
    top1_correct: int
    accepted: int

    @property
    def accuracy(self) -> float:
        return self.accepted_correct / self.total if self.total else 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    manifest_paths: tuple[Path, ...]
    samples: tuple[EvaluationSample, ...]
    results: tuple[SampleEvaluation, ...]

    @property
    def sample_count(self) -> int:
        return len(self.results)

    @property
    def accepted_count(self) -> int:
        return sum(1 for result in self.results if result.accepted)

    @property
    def correct_count(self) -> int:
        return sum(1 for result in self.results if result.correct)

    @property
    def top1_correct_count(self) -> int:
        return sum(1 for result in self.results if result.top1_correct)

    @property
    def error_count(self) -> int:
        return sum(1 for result in self.results if result.error)

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.sample_count if self.sample_count else 0.0

    @property
    def accepted_precision(self) -> float:
        return self.correct_count / self.accepted_count if self.accepted_count else 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct_count / self.sample_count if self.sample_count else 0.0

    @property
    def coverage(self) -> float:
        return self.accepted_count / self.sample_count if self.sample_count else 0.0

    @property
    def p95_ms(self) -> float:
        durations = sorted(result.duration_ms for result in self.results if result.duration_ms >= 0)
        if not durations:
            return 0.0
        if len(durations) < 2:
            return durations[0]
        return quantiles(durations, n=20, method="inclusive")[18]

    @property
    def status_counts(self) -> Counter[str]:
        return Counter(result.status for result in self.results)

    @property
    def confusions(self) -> Counter[tuple[str, str]]:
        counter: Counter[tuple[str, str]] = Counter()
        for result in self.results:
            if result.top1_correct:
                continue
            predicted = result.predicted_species_id or result.status
            counter[(result.sample.expected_species_id, predicted)] += 1
        return counter

    @property
    def species_metrics(self) -> dict[str, SpeciesMetrics]:
        totals: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for result in self.results:
            values = totals[result.sample.expected_species_id]
            values[0] += 1
            if result.correct:
                values[1] += 1
            if result.top1_correct:
                values[2] += 1
            if result.accepted:
                values[3] += 1
        return {
            species_id: SpeciesMetrics(
                total=values[0],
                accepted_correct=values[1],
                top1_correct=values[2],
                accepted=values[3],
            )
            for species_id, values in totals.items()
        }


def ensure_dataset_layout(dataset_root: Path | str) -> tuple[Path, ...]:
    root = Path(dataset_root)
    created: list[Path] = []
    root.mkdir(parents=True, exist_ok=True)
    for stage in DATASET_STAGES:
        stage_dir = root / stage
        if not stage_dir.exists():
            created.append(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
    return tuple(created)


def find_manifest_paths(dataset_root: Path | str, stage: str = "all") -> tuple[Path, ...]:
    root = Path(dataset_root)
    if stage != "all" and stage not in DATASET_STAGES:
        raise ValueError(f"Unknown dataset stage: {stage}")
    stages = DATASET_STAGES if stage == "all" else (stage,)
    manifests: list[Path] = []
    for stage_name in stages:
        stage_dir = root / stage_name
        if stage_dir.exists():
            manifests.extend(sorted(stage_dir.rglob("manifest.json")))
    return tuple(dict.fromkeys(manifests))


def load_evaluation_samples(
    manifest_paths: Iterable[Path | str],
    *,
    include_unapproved: bool = False,
) -> tuple[EvaluationSample, ...]:
    samples: list[EvaluationSample] = []
    for manifest_path_like in manifest_paths:
        manifest_path = Path(manifest_path_like)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in raw.get("samples", []):
            if not isinstance(item, dict):
                continue
            approved = _approved_value(item.get("approved"))
            if approved is False and not include_unapproved:
                continue
            expected = _expected_species_id(item)
            crop_path_text = str(item.get("crop_path", "") or "")
            if not expected or not crop_path_text:
                continue
            sample_id = str(item.get("sample_id") or Path(crop_path_text).stem)
            samples.append(
                EvaluationSample(
                    manifest_path=manifest_path,
                    sample_id=sample_id,
                    crop_path=(manifest_path.parent / crop_path_text),
                    expected_species_id=expected,
                    approved=approved,
                    sample_type=str(item.get("sample_type", "") or ""),
                    notes=str(item.get("notes", "") or ""),
                )
            )
    return tuple(samples)


def evaluate_template_samples(
    samples: Sequence[EvaluationSample],
    matcher: PokemonTemplateMatcher,
    *,
    manifest_paths: Iterable[Path | str] = (),
) -> EvaluationReport:
    results = tuple(_evaluate_one(sample, matcher) for sample in samples)
    return EvaluationReport(
        manifest_paths=tuple(Path(path) for path in manifest_paths),
        samples=tuple(samples),
        results=results,
    )


def evaluate_template_manifests(
    manifest_paths: Iterable[Path | str],
    matcher: PokemonTemplateMatcher,
    *,
    include_unapproved: bool = False,
) -> EvaluationReport:
    paths = tuple(Path(path) for path in manifest_paths)
    samples = load_evaluation_samples(paths, include_unapproved=include_unapproved)
    return evaluate_template_samples(samples, matcher, manifest_paths=paths)


def _evaluate_one(sample: EvaluationSample, matcher: PokemonTemplateMatcher) -> SampleEvaluation:
    if not sample.crop_path.exists():
        return SampleEvaluation(
            sample=sample,
            predicted_species_id=None,
            confidence=0.0,
            accepted=False,
            status="missing-crop",
            correct=False,
            top1_correct=False,
            error=f"Crop image not found: {sample.crop_path}",
        )
    try:
        import time

        started = time.perf_counter()
        match = matcher.match(sample.crop_path.read_bytes())
        duration_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return SampleEvaluation(
            sample=sample,
            predicted_species_id=None,
            confidence=0.0,
            accepted=False,
            status="error",
            correct=False,
            top1_correct=False,
            error=str(exc),
        )
    return _result_from_match(sample, match, duration_ms=duration_ms)


def _result_from_match(sample: EvaluationSample, match: TemplateMatch, *, duration_ms: float = 0.0) -> SampleEvaluation:
    status = _match_status(match)
    top1_correct = match.species_id == sample.expected_species_id
    correct = top1_correct and match.accepted
    return SampleEvaluation(
        sample=sample,
        predicted_species_id=match.species_id,
        confidence=match.confidence,
        accepted=match.accepted,
        status=status,
        correct=correct,
        top1_correct=top1_correct,
        second_species_id=match.second_species_id,
        second_confidence=match.second_confidence,
        duration_ms=duration_ms,
    )


def _match_status(match: TemplateMatch) -> str:
    if match.species_id is None:
        return "no-template"
    if match.accepted:
        return "accepted"
    if match.low_confidence:
        return "low-confidence"
    return "rejected"


def _approved_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _expected_species_id(item: dict[str, object]) -> str:
    for key in ("actual_species_id", "approved_species_id", "species_id", "predicted_species_id"):
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    return ""
