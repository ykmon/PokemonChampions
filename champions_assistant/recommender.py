from __future__ import annotations

from .damage import DamageCalculator
from .data_loader import DataRepository
from .models import BattleSnapshot, FieldSlot, PokemonIdentity, Recommendation, TeamSlot


def build_recommendations(
    snapshot: BattleSnapshot,
    repository: DataRepository,
    language: str = "zh",
) -> list[Recommendation]:
    player_active = [slot for slot in snapshot.player_active if slot.pokemon.is_known]
    opponent_active = [slot for slot in snapshot.opponent_active if slot.pokemon.is_known]
    if not player_active or not opponent_active:
        return [
            Recommendation(
                severity="warning",
                title="场上威胁：需要确认识别结果",
                reason="当前没有稳定识别出双方场上宝可梦。",
                action="先在“当前识别”里手动选择场上槽位，再刷新提示。",
                confidence=0.5,
            )
        ]

    recommendations: list[Recommendation] = []
    recommendations.extend(_field_threats(snapshot, repository, language))
    recommendations.extend(_attack_windows(snapshot, repository, language))
    recommendations.extend(_switch_options(snapshot, repository, language))
    recommendations.extend(_damage_samples(snapshot, repository))
    return recommendations


def _field_threats(
    snapshot: BattleSnapshot,
    repository: DataRepository,
    language: str,
) -> list[Recommendation]:
    chart = repository.type_chart
    recommendations: list[Recommendation] = []
    for opponent_slot in snapshot.opponent_active:
        opponent = opponent_slot.pokemon
        if not opponent.is_known:
            continue
        for player_slot in snapshot.player_active:
            own = player_slot.pokemon
            if not own.is_known:
                continue
            threatening_types = [
                attack_type for attack_type in opponent.types
                if chart.multiplier(attack_type, own.types) > 1
            ]
            if not threatening_types:
                continue
            labels = ", ".join(chart.label(type_name, language) for type_name in threatening_types)
            recommendations.append(
                Recommendation(
                    severity="danger",
                    title="场上威胁",
                    reason=f"{opponent_slot.label} {opponent.name} 的 {labels} 本系攻击克制 {player_slot.label} {own.name}。",
                    action="优先判断是否需要集火、保护或换入抗性/免疫队友。",
                    confidence=min(opponent.confidence, own.confidence),
                )
            )
    return recommendations


def _attack_windows(
    snapshot: BattleSnapshot,
    repository: DataRepository,
    language: str,
) -> list[Recommendation]:
    chart = repository.type_chart
    recommendations: list[Recommendation] = []
    for player_slot in snapshot.player_active:
        own = player_slot.pokemon
        if not own.is_known:
            continue
        for opponent_slot in snapshot.opponent_active:
            opponent = opponent_slot.pokemon
            if not opponent.is_known:
                continue
            best_stab = sorted(
                (
                    (attack_type, chart.multiplier(attack_type, opponent.types))
                    for attack_type in own.types
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if best_stab and best_stab[0][1] > 1:
                attack_type, multiplier = best_stab[0]
                recommendations.append(
                    Recommendation(
                        severity="info",
                        title="进攻机会",
                        reason=f"{player_slot.label} {own.name} 的 {chart.label(attack_type, language)} 本系攻击打 {opponent_slot.label} {opponent.name} 为 x{multiplier:g}。",
                        action="如果速度和血量允许，优先评估该目标是否能确 2 或收割。",
                        confidence=min(own.confidence, opponent.confidence),
                    )
                )
            else:
                profile = chart.defender_profile(opponent.types)
                if profile.weak_to:
                    labels = ", ".join(chart.label(type_name, language) for type_name in profile.weak_to[:4])
                    recommendations.append(
                        Recommendation(
                            severity="info",
                            title="进攻机会",
                            reason=f"{opponent_slot.label} {opponent.name} 弱 {labels}。",
                            action="当前场上若没有对应属性输出，优先看后排或队友补盲。",
                            confidence=0.85,
                        )
                    )
    return _dedupe_recommendations(recommendations, limit=4)


def _switch_options(
    snapshot: BattleSnapshot,
    repository: DataRepository,
    language: str,
) -> list[Recommendation]:
    chart = repository.type_chart
    active_ids = {slot.pokemon.species_id for slot in snapshot.player_active if slot.pokemon.species_id}
    bench = [
        slot for slot in snapshot.player_team
        if slot.pokemon.is_known and slot.pokemon.species_id not in active_ids
    ]
    opponent_attack_types = sorted({
        attack_type
        for opponent_slot in snapshot.opponent_active
        for attack_type in opponent_slot.pokemon.types
        if opponent_slot.pokemon.is_known
    })
    recommendations: list[Recommendation] = []
    for bench_slot in bench:
        pokemon = bench_slot.pokemon
        if not pokemon.is_known:
            continue
        covered = [
            attack_type for attack_type in opponent_attack_types
            if chart.multiplier(attack_type, pokemon.types) < 1
        ]
        immune = [
            attack_type for attack_type in opponent_attack_types
            if chart.multiplier(attack_type, pokemon.types) == 0
        ]
        if not covered:
            continue
        labels = ", ".join(chart.label(type_name, language) for type_name in covered[:4])
        prefix = "免疫" if immune else "抗性"
        recommendations.append(
            Recommendation(
                severity="info",
                title="换入选择",
                reason=f"{bench_slot.label} {pokemon.name} 对当前对方本系压力有{prefix}：{labels}。",
                action="若当前场上被克制，可考虑把它作为换入候选。",
                confidence=0.8,
            )
        )
    return recommendations[:3]


def _damage_samples(snapshot: BattleSnapshot, repository: DataRepository) -> list[Recommendation]:
    calculator = DamageCalculator(repository)
    estimates = calculator.batch_active_estimates(snapshot, side="player", limit=3)
    recommendations: list[Recommendation] = []
    for estimate in estimates:
        attacker_slot = _find_field_slot(snapshot.player_active, estimate.attacker)
        defender_slot = _find_field_slot(snapshot.opponent_active, estimate.defender)
        attacker_label = attacker_slot.label if attacker_slot else "己方场上"
        defender_label = defender_slot.label if defender_slot else "对方场上"
        recommendations.append(
            Recommendation(
                severity="info",
                title="伤害样本",
                reason=(
                    f"{attacker_label} {estimate.attacker.name} -> {defender_label} {estimate.defender.name}: "
                    f"{estimate.move_name} 约 {estimate.percent_min:.1f}%-{estimate.percent_max:.1f}%，属性倍率 x{estimate.type_multiplier:g}。"
                ),
                action="这是基于基础种族值的粗估，实际还要结合性格、努力值、道具和场地修正。",
                confidence=0.75,
            )
        )
    return recommendations


def _find_field_slot(slots: tuple[FieldSlot, ...], pokemon: PokemonIdentity) -> FieldSlot | None:
    for slot in slots:
        if slot.pokemon.species_id == pokemon.species_id:
            return slot
    return None


def _dedupe_recommendations(recommendations: list[Recommendation], limit: int) -> list[Recommendation]:
    seen: set[tuple[str, str]] = set()
    result: list[Recommendation] = []
    for recommendation in recommendations:
        key = (recommendation.title, recommendation.reason)
        if key in seen:
            continue
        seen.add(key)
        result.append(recommendation)
        if len(result) >= limit:
            break
    return result
