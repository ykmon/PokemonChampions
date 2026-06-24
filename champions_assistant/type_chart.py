from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefenderProfile:
    weak_to: tuple[str, ...]
    resists: tuple[str, ...]
    immune_to: tuple[str, ...]


class TypeChart:
    def __init__(
        self,
        types: list[str],
        effectiveness: dict[str, dict[str, float]],
        labels_zh: dict[str, str] | None = None,
    ) -> None:
        self.types = tuple(types)
        self.effectiveness = effectiveness
        self.labels_zh = labels_zh or {}

    def label(self, type_name: str, language: str = "zh") -> str:
        if language == "zh":
            return self.labels_zh.get(type_name, type_name)
        return type_name

    def multiplier(self, attack_type: str, defender_types: tuple[str, ...] | list[str]) -> float:
        if attack_type not in self.types:
            raise KeyError(f"Unknown attacking type: {attack_type}")
        result = 1.0
        chart_row = self.effectiveness.get(attack_type, {})
        for defender_type in defender_types:
            if defender_type not in self.types:
                raise KeyError(f"Unknown defender type: {defender_type}")
            result *= float(chart_row.get(defender_type, 1.0))
        return result

    def defender_profile(self, defender_types: tuple[str, ...] | list[str]) -> DefenderProfile:
        weak: list[str] = []
        resists: list[str] = []
        immune: list[str] = []
        for attack_type in self.types:
            multiplier = self.multiplier(attack_type, defender_types)
            if multiplier == 0:
                immune.append(attack_type)
            elif multiplier > 1:
                weak.append(attack_type)
            elif multiplier < 1:
                resists.append(attack_type)
        return DefenderProfile(tuple(weak), tuple(resists), tuple(immune))

    def best_attack_types(self, defender_types: tuple[str, ...] | list[str]) -> tuple[tuple[str, float], ...]:
        ranked = [(attack_type, self.multiplier(attack_type, defender_types)) for attack_type in self.types]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return tuple(item for item in ranked if item[1] > 1)
