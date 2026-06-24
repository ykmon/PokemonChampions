from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


MoveCategory = Literal["physical", "special", "status"]
BattleSide = Literal["player", "opponent"]


class BattleFormat(str, Enum):
    SINGLES_63 = "singles63"
    DOUBLES_64 = "doubles64"

    @property
    def active_slots_per_side(self) -> int:
        return 1 if self == BattleFormat.SINGLES_63 else 2

    @property
    def selected_team_size(self) -> int:
        return 3 if self == BattleFormat.SINGLES_63 else 4

    @property
    def label_zh(self) -> str:
        return "63 单打" if self == BattleFormat.SINGLES_63 else "64 双打"

    @classmethod
    def parse(cls, value: str | "BattleFormat") -> "BattleFormat":
        if isinstance(value, BattleFormat):
            return value
        normalized = value.strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "63": cls.SINGLES_63,
            "single": cls.SINGLES_63,
            "singles": cls.SINGLES_63,
            "singles63": cls.SINGLES_63,
            "64": cls.DOUBLES_64,
            "double": cls.DOUBLES_64,
            "doubles": cls.DOUBLES_64,
            "doubles64": cls.DOUBLES_64,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown battle format: {value}")


@dataclass(frozen=True)
class Rect:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "Rect":
        values = values or {}
        return cls(
            x=int(values.get("x", 0) or 0),
            y=int(values.get("y", 0) or 0),
            width=int(values.get("width", 0) or 0),
            height=int(values.get("height", 0) or 0),
        )

    @property
    def enabled(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def clamp(self, image_width: int, image_height: int) -> "Rect":
        x = max(0, min(self.x, image_width))
        y = max(0, min(self.y, image_height))
        width = max(0, min(self.width, image_width - x))
        height = max(0, min(self.height, image_height - y))
        return Rect(x=x, y=y, width=width, height=height)


@dataclass(frozen=True)
class PokemonIdentity:
    name: str = "Unknown"
    species_id: str | None = None
    form: str | None = None
    types: tuple[str, ...] = ()
    confidence: float = 0.0
    source: str = "unknown"

    @property
    def is_known(self) -> bool:
        return self.species_id is not None and bool(self.types)

    @property
    def is_identified(self) -> bool:
        return self.species_id is not None


@dataclass(frozen=True)
class TeamSlot:
    side: BattleSide
    index: int
    pokemon: PokemonIdentity = field(default_factory=PokemonIdentity)
    selected: bool = False
    locked: bool = False

    @property
    def label(self) -> str:
        prefix = "己方队伍" if self.side == "player" else "对方队伍"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class FieldSlot:
    side: BattleSide
    index: int
    pokemon: PokemonIdentity = field(default_factory=PokemonIdentity)
    hp_text: str = ""
    status_text: str = ""
    team_slot_index: int | None = None
    locked: bool = False

    @property
    def label(self) -> str:
        prefix = "己方场上" if self.side == "player" else "对方场上"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Move:
    name: str
    move_type: str
    category: MoveCategory
    power: int
    accuracy: int = 100
    name_zh: str = ""

    @property
    def is_damaging(self) -> bool:
        return self.category in {"physical", "special"} and self.power > 0


@dataclass(frozen=True)
class BattleSnapshot:
    battle_format: BattleFormat = BattleFormat.SINGLES_63
    player_team: tuple[TeamSlot, ...] = field(default_factory=lambda: make_team_slots("player"))
    opponent_team: tuple[TeamSlot, ...] = field(default_factory=lambda: make_team_slots("opponent"))
    player_active: tuple[FieldSlot, ...] = field(default_factory=lambda: make_field_slots("player", BattleFormat.SINGLES_63))
    opponent_active: tuple[FieldSlot, ...] = field(default_factory=lambda: make_field_slots("opponent", BattleFormat.SINGLES_63))
    turn_text: str = ""
    source_image: str = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def empty(cls, battle_format: BattleFormat | str = BattleFormat.SINGLES_63) -> "BattleSnapshot":
        fmt = BattleFormat.parse(battle_format)
        return cls(
            battle_format=fmt,
            player_team=make_team_slots("player"),
            opponent_team=make_team_slots("opponent"),
            player_active=make_field_slots("player", fmt),
            opponent_active=make_field_slots("opponent", fmt),
        )

    @classmethod
    def from_pair(
        cls,
        self_pokemon: PokemonIdentity,
        opponent_pokemon: PokemonIdentity,
        battle_format: BattleFormat | str = BattleFormat.SINGLES_63,
    ) -> "BattleSnapshot":
        fmt = BattleFormat.parse(battle_format)
        return cls.empty(fmt).with_active_pair(self_pokemon, opponent_pokemon)

    @property
    def active_slots_per_side(self) -> int:
        return self.battle_format.active_slots_per_side

    @property
    def self_pokemon(self) -> PokemonIdentity:
        return self.player_active[0].pokemon if self.player_active else PokemonIdentity()

    @property
    def opponent_pokemon(self) -> PokemonIdentity:
        return self.opponent_active[0].pokemon if self.opponent_active else PokemonIdentity()

    @property
    def self_hp_text(self) -> str:
        return self.player_active[0].hp_text if self.player_active else ""

    @property
    def opponent_hp_text(self) -> str:
        return self.opponent_active[0].hp_text if self.opponent_active else ""

    def with_format(self, battle_format: BattleFormat | str) -> "BattleSnapshot":
        fmt = BattleFormat.parse(battle_format)
        return replace(
            self,
            battle_format=fmt,
            player_active=_resize_field_slots(self.player_active, "player", fmt),
            opponent_active=_resize_field_slots(self.opponent_active, "opponent", fmt),
        )

    def with_active_pair(self, player: PokemonIdentity, opponent: PokemonIdentity) -> "BattleSnapshot":
        player_active = update_field_slot(self.player_active, 1, player)
        opponent_active = update_field_slot(self.opponent_active, 1, opponent)
        player_team = update_team_slot(self.player_team, 1, player, selected=True)
        opponent_team = update_team_slot(self.opponent_team, 1, opponent, selected=True)
        return replace(
            self,
            player_active=player_active,
            opponent_active=opponent_active,
            player_team=player_team,
            opponent_team=opponent_team,
        )

    def active_pokemon(self, side: BattleSide) -> tuple[PokemonIdentity, ...]:
        slots = self.player_active if side == "player" else self.opponent_active
        return tuple(slot.pokemon for slot in slots if slot.pokemon.is_known)

    def team_pokemon(self, side: BattleSide) -> tuple[PokemonIdentity, ...]:
        slots = self.player_team if side == "player" else self.opponent_team
        return tuple(slot.pokemon for slot in slots if slot.pokemon.is_known)


def make_team_slots(side: BattleSide, count: int = 6) -> tuple[TeamSlot, ...]:
    return tuple(TeamSlot(side=side, index=index) for index in range(1, count + 1))


def make_field_slots(side: BattleSide, battle_format: BattleFormat | str) -> tuple[FieldSlot, ...]:
    fmt = BattleFormat.parse(battle_format)
    return tuple(FieldSlot(side=side, index=index) for index in range(1, fmt.active_slots_per_side + 1))


def update_team_slot(
    slots: tuple[TeamSlot, ...],
    index: int,
    pokemon: PokemonIdentity,
    *,
    selected: bool | None = None,
    locked: bool | None = None,
) -> tuple[TeamSlot, ...]:
    updated: list[TeamSlot] = []
    for slot in slots:
        if slot.index == index:
            updated.append(
                replace(
                    slot,
                    pokemon=pokemon,
                    selected=slot.selected if selected is None else selected,
                    locked=slot.locked if locked is None else locked,
                )
            )
        else:
            updated.append(slot)
    return tuple(updated)


def update_field_slot(
    slots: tuple[FieldSlot, ...],
    index: int,
    pokemon: PokemonIdentity,
    *,
    hp_text: str | None = None,
    status_text: str | None = None,
    team_slot_index: int | None = None,
    locked: bool | None = None,
) -> tuple[FieldSlot, ...]:
    updated: list[FieldSlot] = []
    for slot in slots:
        if slot.index == index:
            updated.append(
                replace(
                    slot,
                    pokemon=pokemon,
                    hp_text=slot.hp_text if hp_text is None else hp_text,
                    status_text=slot.status_text if status_text is None else status_text,
                    team_slot_index=slot.team_slot_index if team_slot_index is None else team_slot_index,
                    locked=slot.locked if locked is None else locked,
                )
            )
        else:
            updated.append(slot)
    return tuple(updated)


def merge_identity(existing: PokemonIdentity, recognized: PokemonIdentity) -> PokemonIdentity:
    if recognized.is_identified:
        return recognized
    return existing if existing.is_identified else recognized


def _resize_field_slots(
    slots: tuple[FieldSlot, ...],
    side: BattleSide,
    battle_format: BattleFormat,
) -> tuple[FieldSlot, ...]:
    needed = battle_format.active_slots_per_side
    if len(slots) == needed:
        return slots
    resized = list(slots[:needed])
    while len(resized) < needed:
        resized.append(FieldSlot(side=side, index=len(resized) + 1))
    return tuple(resized)


@dataclass(frozen=True)
class DamageEstimate:
    move_name: str
    attacker: PokemonIdentity
    defender: PokemonIdentity
    attack_type: str
    type_multiplier: float
    stab: bool
    damage_min: int
    damage_max: int
    percent_min: float
    percent_max: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    severity: Literal["info", "warning", "danger"]
    title: str
    reason: str
    action: str
    confidence: float = 1.0
