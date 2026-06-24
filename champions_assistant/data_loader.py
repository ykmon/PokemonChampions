from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import Move, PokemonIdentity
from .paths import DEFAULT_DATA_DIR
from .type_chart import TypeChart


_SPACE_RE = re.compile(r"\s+")


class DataRepository:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.metadata = _read_json(self.data_dir / "metadata.json")
        self._pokemon_raw = _read_json(self.data_dir / "pokemon.json").get("pokemon", [])
        self._moves_raw = _read_json(self.data_dir / "moves.json").get("moves", [])
        aliases_raw = _read_json(self.data_dir / "aliases.json").get("aliases", {})
        chart_raw = _read_json(self.data_dir / "type_chart.json")

        self.type_chart = TypeChart(
            types=chart_raw["types"],
            effectiveness=chart_raw["effectiveness"],
            labels_zh=chart_raw.get("labels_zh", {}),
        )
        self.pokemon_by_id = {entry["id"]: entry for entry in self._pokemon_raw}
        self.moves_by_name = {entry["name"]: _move_from_raw(entry) for entry in self._moves_raw}
        self.aliases = {_normalize(alias): species_id for alias, species_id in aliases_raw.items()}
        for species_id, entry in self.pokemon_by_id.items():
            self.aliases.setdefault(_normalize(species_id), species_id)
            self.aliases.setdefault(_normalize(entry["name"]), species_id)
            if entry.get("name_zh"):
                self.aliases.setdefault(_normalize(entry["name_zh"]), species_id)

    def resolve_pokemon(self, query: str, confidence: float = 1.0, source: str = "manual") -> PokemonIdentity:
        species_id = self.match_species_id(query)
        if species_id is None:
            return PokemonIdentity(name=query.strip() or "Unknown", confidence=0.0, source=source)
        return self.identity_for_id(species_id, confidence=confidence, source=source)

    def match_species_id(self, query: str) -> str | None:
        normalized = _normalize(query)
        if not normalized:
            return None
        if normalized in self.aliases:
            return self.aliases[normalized]

        compact = normalized.replace(" ", "")
        for alias, species_id in self.aliases.items():
            if alias.replace(" ", "") == compact:
                return species_id
        for alias, species_id in self.aliases.items():
            if compact in alias.replace(" ", "") or alias.replace(" ", "") in compact:
                return species_id
        return None

    def identity_for_id(self, species_id: str, confidence: float = 1.0, source: str = "data") -> PokemonIdentity:
        entry = self.pokemon_by_id[species_id]
        return PokemonIdentity(
            name=entry["name"],
            species_id=species_id,
            types=tuple(entry["types"]),
            confidence=confidence,
            source=source,
        )

    def pokemon_label(self, species_id: str, language: str = "zh") -> str:
        entry = self.pokemon_by_id[species_id]
        if language == "zh" and entry.get("name_zh"):
            return f'{entry["name_zh"]} / {entry["name"]}'
        return entry["name"]

    def all_pokemon(self) -> list[PokemonIdentity]:
        return [self.identity_for_id(species_id) for species_id in sorted(self.pokemon_by_id)]

    def base_stats(self, species_id: str | None) -> dict[str, int]:
        if not species_id or species_id not in self.pokemon_by_id:
            return {"hp": 80, "attack": 100, "defense": 100, "sp_attack": 100, "sp_defense": 100, "speed": 80}
        return dict(self.pokemon_by_id[species_id].get("base_stats", {}))

    def moves_for_pokemon(self, pokemon: PokemonIdentity) -> list[Move]:
        if not pokemon.species_id or pokemon.species_id not in self.pokemon_by_id:
            return []
        entry = self.pokemon_by_id[pokemon.species_id]
        moves: list[Move] = []
        for move_name in entry.get("common_moves", []):
            move = self.moves_by_name.get(move_name)
            if move:
                moves.append(move)
        return moves

    def damaging_moves_for_pokemon(self, pokemon: PokemonIdentity) -> list[Move]:
        return [move for move in self.moves_for_pokemon(pokemon) if move.is_damaging]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _move_from_raw(raw: dict[str, Any]) -> Move:
    return Move(
        name=raw["name"],
        name_zh=raw.get("name_zh", ""),
        move_type=raw["type"],
        category=raw["category"],
        power=int(raw.get("power", 0) or 0),
        accuracy=int(raw.get("accuracy", 100) or 100),
    )


def _normalize(value: str) -> str:
    value = value.strip().casefold()
    return _SPACE_RE.sub(" ", value)
