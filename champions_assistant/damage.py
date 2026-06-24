from __future__ import annotations

from .data_loader import DataRepository
from .models import BattleSnapshot, DamageEstimate, Move, PokemonIdentity


class DamageCalculator:
    def __init__(self, repository: DataRepository, level: int = 50) -> None:
        self.repository = repository
        self.level = level

    def estimate(
        self,
        attacker: PokemonIdentity,
        defender: PokemonIdentity,
        move: Move,
        defender_hp: int | None = None,
    ) -> DamageEstimate:
        defender_hp = defender_hp or self._hp_stat(defender)
        notes: list[str] = []
        type_multiplier = self.repository.type_chart.multiplier(move.move_type, defender.types)
        stab = move.move_type in attacker.types

        if not move.is_damaging or type_multiplier == 0:
            reason = "status move" if not move.is_damaging else "type immunity"
            notes.append(reason)
            return DamageEstimate(
                move_name=move.name,
                attacker=attacker,
                defender=defender,
                attack_type=move.move_type,
                type_multiplier=type_multiplier,
                stab=stab,
                damage_min=0,
                damage_max=0,
                percent_min=0,
                percent_max=0,
                notes=tuple(notes),
            )

        attacker_stats = self.repository.base_stats(attacker.species_id)
        defender_stats = self.repository.base_stats(defender.species_id)
        attack_stat = attacker_stats.get("attack" if move.category == "physical" else "sp_attack", 100)
        defense_stat = defender_stats.get("defense" if move.category == "physical" else "sp_defense", 100)
        base = (((2 * self.level / 5 + 2) * move.power * attack_stat / max(1, defense_stat)) / 50) + 2
        modifier = type_multiplier * (1.5 if stab else 1.0)
        max_damage = int(base * modifier)
        min_damage = int(base * modifier * 0.85)
        if stab:
            notes.append("STAB")
        if type_multiplier > 1:
            notes.append(f"super effective x{type_multiplier:g}")
        elif 0 < type_multiplier < 1:
            notes.append(f"resisted x{type_multiplier:g}")

        return DamageEstimate(
            move_name=move.name,
            attacker=attacker,
            defender=defender,
            attack_type=move.move_type,
            type_multiplier=type_multiplier,
            stab=stab,
            damage_min=max(1, min_damage),
            damage_max=max(1, max_damage),
            percent_min=round(max(0, min_damage) / defender_hp * 100, 1),
            percent_max=round(max(0, max_damage) / defender_hp * 100, 1),
            notes=tuple(notes),
        )

    def best_estimates(self, attacker: PokemonIdentity, defender: PokemonIdentity, limit: int = 4) -> list[DamageEstimate]:
        estimates = [
            self.estimate(attacker, defender, move)
            for move in self.repository.damaging_moves_for_pokemon(attacker)
        ]
        estimates.sort(key=lambda estimate: (estimate.percent_max, estimate.type_multiplier), reverse=True)
        return estimates[:limit]

    def batch_active_estimates(
        self,
        snapshot: BattleSnapshot,
        *,
        side: str = "player",
        limit: int = 8,
    ) -> list[DamageEstimate]:
        if side == "player":
            attackers = [slot.pokemon for slot in snapshot.player_active if slot.pokemon.is_known]
            defenders = [slot.pokemon for slot in snapshot.opponent_active if slot.pokemon.is_known]
        else:
            attackers = [slot.pokemon for slot in snapshot.opponent_active if slot.pokemon.is_known]
            defenders = [slot.pokemon for slot in snapshot.player_active if slot.pokemon.is_known]

        estimates: list[DamageEstimate] = []
        for attacker in attackers:
            for defender in defenders:
                estimates.extend(self.best_estimates(attacker, defender, limit=4))
        estimates.sort(key=lambda estimate: (estimate.percent_max, estimate.type_multiplier), reverse=True)
        return estimates[:limit]

    def _hp_stat(self, defender: PokemonIdentity) -> int:
        stats = self.repository.base_stats(defender.species_id)
        base_hp = stats.get("hp", 80)
        return int(((2 * base_hp + 31) * self.level / 100) + self.level + 10)
