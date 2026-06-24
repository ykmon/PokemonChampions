from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adb import AdbClient, AdbError
from .config import load_config
from .damage import DamageCalculator
from .data_loader import DataRepository
from .models import BattleFormat, BattleSnapshot, update_field_slot, update_team_slot
from .preview_recognition import recognize_opponent_preview
from .recommender import build_recommendations
from .roi import VisionDependencyError
from .templates import PokemonTemplateMatcher, crop_image_bytes, default_opponent_preview_rois_1920


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="champions-assistant")
    parser.add_argument("--config", default="config/app.toml", help="Path to app TOML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Start the desktop assistant.")
    subparsers.add_parser("calibrate", help="Open the ROI calibration window.")

    capture = subparsers.add_parser("capture", help="Save one emulator screenshot through ADB.")
    capture.add_argument("--out", default="screenshots/sample.png", help="PNG output path.")

    harvest = subparsers.add_parser("harvest-templates", help="Crop opponent preview icons and save them as templates.")
    harvest.add_argument("--image", required=True, help="Team preview screenshot path.")
    harvest.add_argument("--opponent", required=True, help="Comma-separated opponent species ids or aliases, top to bottom.")

    recognize = subparsers.add_parser("recognize-preview", help="Recognize opponent preview icons from a screenshot.")
    recognize.add_argument("--image", required=True, help="Team preview screenshot path.")

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
    if args.command == "harvest-templates":
        return _harvest_templates(config, Path(args.image), args.opponent)
    if args.command == "recognize-preview":
        return _recognize_preview(config, Path(args.image))
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
    client = AdbClient(config.adb_path, config.device_serial)
    try:
        client.capture_screenshot(out_path)
    except AdbError as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved screenshot to {out_path}")
    return 0


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
    matcher = PokemonTemplateMatcher(repository)
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
                    f"template={result.template_path}"
                )
            else:
                print(f"{result.roi_key}: Unknown confidence=0.000 status=no-template")
    except (VisionDependencyError, ValueError) as exc:
        print(f"Preview recognition failed: {exc}", file=sys.stderr)
        return 1
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
