from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adb import AdbError
from .config import load_config
from .damage import DamageCalculator
from .data_loader import DataRepository
from .emulator import adb_client_from_config
from .evaluation import (
    DATASET_STAGES,
    ensure_dataset_layout,
    evaluate_template_manifests,
    find_manifest_paths,
)
from .health import build_health_report
from .models import BattleFormat, BattleSnapshot, update_field_slot, update_team_slot
from .preview_recognition import recognize_opponent_preview
from .recommender import build_recommendations
from .roi import VisionDependencyError
from .templates import _cv2, _np, crop_image_bytes, default_opponent_preview_rois_1920
from .vision_config import build_template_matcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="champions-assistant")
    parser.add_argument("--config", default="config/app.toml", help="Path to app TOML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Start the desktop assistant.")
    subparsers.add_parser("calibrate", help="Open the ROI calibration window.")

    capture = subparsers.add_parser("capture", help="Save one emulator screenshot through ADB.")
    capture.add_argument("--out", default="screenshots/sample.png", help="PNG output path.")

    subparsers.add_parser("capture-benchmark", help="Benchmark ADB screenshot methods and select the fastest usable one.")

    harvest = subparsers.add_parser("harvest-templates", help="Crop opponent preview icons and save them as templates.")
    harvest.add_argument("--image", required=True, help="Team preview screenshot path.")
    harvest.add_argument("--opponent", required=True, help="Comma-separated opponent species ids or aliases, top to bottom.")

    recognize = subparsers.add_parser("recognize-preview", help="Recognize opponent preview icons from a screenshot.")
    recognize.add_argument("--image", required=True, help="Team preview screenshot path.")

    evaluate = subparsers.add_parser("evaluate-templates", help="Evaluate template matching against dataset manifests.")
    evaluate.add_argument("--dataset", default="dataset", help="Dataset root containing lifecycle stage directories.")
    evaluate.add_argument(
        "--stage",
        choices=["all", *DATASET_STAGES],
        default="all",
        help="Dataset stage to scan for manifest.json files.",
    )
    evaluate.add_argument("--manifest", action="append", default=[], help="Specific manifest path. May be repeated.")
    evaluate.add_argument("--include-unapproved", action="store_true", help="Include samples explicitly marked approved=false.")
    evaluate.add_argument("--limit-confusions", type=int, default=10, help="Number of confusion pairs to print.")
    evaluate.add_argument("--limit-species", type=int, default=10, help="Number of weakest species rows to print.")

    subparsers.add_parser("health-check", help="Check local dependencies, data, ADB, ROI, and template readiness.")

    init_dataset = subparsers.add_parser("init-dataset-layout", help="Create dataset lifecycle stage directories.")
    init_dataset.add_argument("--dataset", default="dataset", help="Dataset root to initialize.")

    analyze = subparsers.add_parser("analyze", help="Analyze a manually selected matchup.")
    analyze.add_argument("--format", choices=[item.value for item in BattleFormat], default="singles63")
    analyze.add_argument("--self", dest="self_name", help="Legacy: your active Pokemon name or alias.")
    analyze.add_argument("--opponent", help="Legacy: opponent active Pokemon name or alias.")
    analyze.add_argument("--self-team", default="", help="Comma-separated six Pokemon aliases for your team.")
    analyze.add_argument("--opponent-team", default="", help="Comma-separated six Pokemon aliases for opponent team.")
    analyze.add_argument("--self-active", default="", help="Comma-separated active Pokemon aliases. One for 63, up to two for 64.")
    analyze.add_argument("--opponent-active", default="", help="Comma-separated active opponent aliases. One for 63, up to two for 64.")
    analyze.add_argument("--move", default="", help="Optional move name for a damage estimate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "run":
        return _run_ui(config)
    if args.command == "calibrate":
        return _run_calibration(config, Path(args.config))
    if args.command == "capture":
        return _capture(config, Path(args.out))
    if args.command == "capture-benchmark":
        return _capture_benchmark(config)
    if args.command == "harvest-templates":
        return _harvest_templates(config, Path(args.image), args.opponent)
    if args.command == "recognize-preview":
        return _recognize_preview(config, Path(args.image))
    if args.command == "evaluate-templates":
        return _evaluate_templates(
            config,
            dataset_root=Path(args.dataset),
            stage=args.stage,
            manifest_paths=[Path(path) for path in args.manifest],
            include_unapproved=args.include_unapproved,
            limit_confusions=args.limit_confusions,
            limit_species=args.limit_species,
        )
    if args.command == "health-check":
        return _health_check(config)
    if args.command == "init-dataset-layout":
        return _init_dataset_layout(Path(args.dataset))
    if args.command == "analyze":
        return _analyze(
            config,
            args.self_name,
            args.opponent,
            args.move,
            args.format,
            args.self_team,
            args.opponent_team,
            args.self_active,
            args.opponent_active,
        )
    return 2


def _run_ui(config) -> int:
    try:
        from .ui import run_app
    except ImportError as exc:
        print("PySide6 is required for the desktop UI. Install with: python -m pip install -e .[ui]", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 2
    return run_app(config)


def _run_calibration(config, config_path: Path) -> int:
    try:
        from .ui import run_calibration
    except ImportError as exc:
        print("PySide6 is required for calibration. Install with: python -m pip install -e .[ui]", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 2
    return run_calibration(config, config_path)


def _capture(config, out_path: Path) -> int:
    client = adb_client_from_config(config)
    try:
        client.capture_screenshot(out_path)
    except AdbError as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved screenshot to {out_path}")
    if client.capture_benchmark is not None:
        print(f"Capture method: {client.capture_benchmark.summary()}")
    return 0


def _capture_benchmark(config) -> int:
    client = adb_client_from_config(config)
    try:
        benchmark = client.benchmark_capture_methods()
    except AdbError as exc:
        print(f"Capture benchmark failed: {exc}", file=sys.stderr)
        return 1
    print("Capture benchmark")
    print(f"  {benchmark.summary()}")
    return 0 if benchmark.ok else 1


def _harvest_templates(config, image_path: Path, opponent: str) -> int:
    repository = DataRepository(config.data_dir)
    names = _split_names(opponent)
    if len(names) != 6:
        print("--opponent must contain exactly 6 Pokemon names/species ids, top to bottom.", file=sys.stderr)
        return 2
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1
    image_bytes = image_path.read_bytes()
    matcher = build_template_matcher(repository)
    rois = _opponent_preview_rois(config)
    try:
        for index, name in enumerate(names, start=1):
            identity = repository.resolve_pokemon(name)
            if not identity.species_id:
                print(f"Unknown Pokemon for slot {index}: {name}", file=sys.stderr)
                return 1
            crop = crop_image_bytes(image_bytes, rois[f"opponent_preview_{index}"])
            saved = matcher.save_template(identity.species_id, crop)
            print(f"slot {index}: {identity.name} -> {saved}")
    except (VisionDependencyError, ValueError, KeyError) as exc:
        print(f"Template harvest failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _recognize_preview(config, image_path: Path) -> int:
    repository = DataRepository(config.data_dir)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1
    image_bytes = image_path.read_bytes()
    try:
        results = recognize_opponent_preview(config, repository, image_bytes)
        for result in results:
            if result.species_id:
                print(
                    f"{result.roi_key}: {result.label} "
                    f"confidence={result.confidence:.3f} "
                    f"status={result.status} "
                    f"ms={result.elapsed_ms:.2f} "
                    f"failure={result.failure_reason or '-'} "
                    f"template={result.template_path}"
                )
            else:
                print(
                    f"{result.roi_key}: Unknown confidence=0.000 "
                    f"status={result.status} ms={result.elapsed_ms:.2f} "
                    f"failure={result.failure_reason or '-'}"
                )
    except (VisionDependencyError, ValueError) as exc:
        print(f"Preview recognition failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _evaluate_templates(
    config,
    *,
    dataset_root: Path,
    stage: str,
    manifest_paths: list[Path],
    include_unapproved: bool,
    limit_confusions: int,
    limit_species: int,
) -> int:
    repository = DataRepository(config.data_dir)
    paths = tuple(manifest_paths) if manifest_paths else find_manifest_paths(dataset_root, stage)
    if not paths:
        print(f"No manifest.json files found for stage '{stage}' under {dataset_root}.", file=sys.stderr)
        return 1
    matcher = build_template_matcher(repository)
    try:
        _cv2()
        _np()
        report = evaluate_template_manifests(paths, matcher, include_unapproved=include_unapproved)
    except (OSError, ValueError, VisionDependencyError) as exc:
        print(f"Template evaluation failed: {exc}", file=sys.stderr)
        return 1
    if report.sample_count == 0:
        print("No labeled samples found. Use --include-unapproved for pending review manifests.", file=sys.stderr)
        return 1

    print("Template evaluation")
    print(f"  manifests: {len(report.manifest_paths)}")
    print(f"  samples: {report.sample_count}")
    print(f"  accepted coverage: {_percent(report.coverage)} ({report.accepted_count}/{report.sample_count})")
    print(f"  accepted precision: {_percent(report.accepted_precision)} ({report.correct_count}/{report.accepted_count})")
    print(f"  accepted accuracy: {_percent(report.accuracy)} ({report.correct_count}/{report.sample_count})")
    print(f"  top-1 accuracy before threshold: {_percent(report.top1_accuracy)} ({report.top1_correct_count}/{report.sample_count})")
    print(f"  p95 match time: {report.p95_ms:.2f} ms")
    if report.error_count:
        print(f"  errors: {report.error_count}")
    print("  statuses:")
    for status, count in sorted(report.status_counts.items()):
        print(f"    {status}: {count}")

    confusions = report.confusions.most_common(max(0, limit_confusions))
    if confusions:
        print("  confusions:")
        for (expected, predicted), count in confusions:
            print(f"    {expected} -> {predicted}: {count}")

    weakest = sorted(
        report.species_metrics.items(),
        key=lambda item: (item[1].accuracy, item[1].top1_accuracy, -item[1].total),
    )[:max(0, limit_species)]
    if weakest:
        print("  weakest species:")
        for species_id, metrics in weakest:
            print(
                f"    {species_id}: accepted={_percent(metrics.accuracy)} "
                f"top1={_percent(metrics.top1_accuracy)} "
                f"accepted_count={metrics.accepted}/{metrics.total}"
            )
    return 0 if report.error_count == 0 else 1


def _health_check(config) -> int:
    report = build_health_report(config)
    print("\n".join(report.lines()))
    return 0 if report.blocking_ok else 1


def _init_dataset_layout(dataset_root: Path) -> int:
    created = ensure_dataset_layout(dataset_root)
    print(f"Dataset layout ready: {dataset_root}")
    for stage in DATASET_STAGES:
        marker = "created" if dataset_root / stage in created else "exists"
        print(f"  {marker}: {dataset_root / stage}")
    return 0


def _analyze(
    config,
    self_name: str | None,
    opponent_name: str | None,
    move_name: str = "",
    battle_format: str = "singles63",
    self_team: str = "",
    opponent_team: str = "",
    self_active: str = "",
    opponent_active: str = "",
) -> int:
    repository = DataRepository(config.data_dir)
    snapshot = _snapshot_from_cli(
        repository,
        battle_format=battle_format,
        self_name=self_name,
        opponent_name=opponent_name,
        self_team=self_team,
        opponent_team=opponent_team,
        self_active=self_active,
        opponent_active=opponent_active,
    )
    for recommendation in build_recommendations(snapshot, repository, config.language):
        print(f"[{recommendation.severity}] {recommendation.title}")
        print(f"  {recommendation.reason}")
        print(f"  {recommendation.action}")

    if move_name:
        move = repository.moves_by_name.get(move_name)
        if move is None:
            print(f"Unknown move: {move_name}", file=sys.stderr)
            return 1
        estimate = DamageCalculator(repository).estimate(snapshot.self_pokemon, snapshot.opponent_pokemon, move)
        print(
            f"{estimate.move_name}: {estimate.damage_min}-{estimate.damage_max} "
            f"({estimate.percent_min:.1f}%-{estimate.percent_max:.1f}%), x{estimate.type_multiplier:g}"
        )
    return 0


def _snapshot_from_cli(
    repository: DataRepository,
    *,
    battle_format: str,
    self_name: str | None,
    opponent_name: str | None,
    self_team: str,
    opponent_team: str,
    self_active: str,
    opponent_active: str,
) -> BattleSnapshot:
    fmt = BattleFormat.parse(battle_format)
    snapshot = BattleSnapshot.empty(fmt)

    player_team_names = _split_names(self_team)
    opponent_team_names = _split_names(opponent_team)
    player_active_names = _split_names(self_active)
    opponent_active_names = _split_names(opponent_active)

    if self_name and not player_active_names:
        player_active_names = [self_name]
    if opponent_name and not opponent_active_names:
        opponent_active_names = [opponent_name]
    if not player_team_names:
        player_team_names = player_active_names
    if not opponent_team_names:
        opponent_team_names = opponent_active_names

    for index, name in enumerate(player_team_names[:6], start=1):
        snapshot = _with_team_pokemon(snapshot, repository, "player", index, name)
    for index, name in enumerate(opponent_team_names[:6], start=1):
        snapshot = _with_team_pokemon(snapshot, repository, "opponent", index, name)
    for index, name in enumerate(player_active_names[:fmt.active_slots_per_side], start=1):
        snapshot = _with_active_pokemon(snapshot, repository, "player", index, name)
    for index, name in enumerate(opponent_active_names[:fmt.active_slots_per_side], start=1):
        snapshot = _with_active_pokemon(snapshot, repository, "opponent", index, name)

    return snapshot


def _with_team_pokemon(snapshot: BattleSnapshot, repository: DataRepository, side: str, index: int, name: str) -> BattleSnapshot:
    pokemon = repository.resolve_pokemon(name)
    if side == "player":
        return BattleSnapshot(
            battle_format=snapshot.battle_format,
            player_team=update_team_slot(snapshot.player_team, index, pokemon, selected=index <= snapshot.battle_format.selected_team_size),
            opponent_team=snapshot.opponent_team,
            player_active=snapshot.player_active,
            opponent_active=snapshot.opponent_active,
            turn_text=snapshot.turn_text,
            source_image=snapshot.source_image,
        )
    return BattleSnapshot(
        battle_format=snapshot.battle_format,
        player_team=snapshot.player_team,
        opponent_team=update_team_slot(snapshot.opponent_team, index, pokemon, selected=index <= snapshot.battle_format.selected_team_size),
        player_active=snapshot.player_active,
        opponent_active=snapshot.opponent_active,
        turn_text=snapshot.turn_text,
        source_image=snapshot.source_image,
    )


def _with_active_pokemon(snapshot: BattleSnapshot, repository: DataRepository, side: str, index: int, name: str) -> BattleSnapshot:
    pokemon = repository.resolve_pokemon(name)
    if side == "player":
        return BattleSnapshot(
            battle_format=snapshot.battle_format,
            player_team=snapshot.player_team,
            opponent_team=snapshot.opponent_team,
            player_active=update_field_slot(snapshot.player_active, index, pokemon),
            opponent_active=snapshot.opponent_active,
            turn_text=snapshot.turn_text,
            source_image=snapshot.source_image,
        )
    return BattleSnapshot(
        battle_format=snapshot.battle_format,
        player_team=snapshot.player_team,
        opponent_team=snapshot.opponent_team,
        player_active=snapshot.player_active,
        opponent_active=update_field_slot(snapshot.opponent_active, index, pokemon),
        turn_text=snapshot.turn_text,
        source_image=snapshot.source_image,
    )


def _split_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _opponent_preview_rois(config) -> dict[str, object]:
    defaults = default_opponent_preview_rois_1920()
    rois = {}
    for key, default_rect in defaults.items():
        configured = config.rois.get(key)
        rois[key] = configured if configured and configured.enabled else default_rect
    return rois


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"
