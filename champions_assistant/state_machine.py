from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import BattleSnapshot
from .preview_recognition import PreviewRecognitionResult, accepted_count


class AssistantState(str, Enum):
    UNKNOWN = "unknown"
    DETECT_TEAM_PREVIEW = "detect_team_preview"
    ANALYZE_MATCHUP = "analyze_matchup"
    WAIT_BATTLE = "wait_battle"
    DETECT_ACTIVE = "detect_active"
    RECOMMEND = "recommend"
    ERROR = "error"

    # Backward-compatible aliases for the v0 state names.
    TEAM_PREVIEW = "detect_team_preview"
    BATTLE_ACTIVE = "detect_active"
    ANALYSIS_READY = "recommend"


@dataclass(frozen=True)
class StateMachineResult:
    state: AssistantState
    screen: str
    should_refresh_recommendations: bool
    message: str = ""
    diagnostics: tuple[str, ...] = ()

    @property
    def refresh_recommendations(self) -> bool:
        return self.should_refresh_recommendations


def evaluate_readonly_state(
    snapshot: BattleSnapshot | None = None,
    preview_results: tuple[PreviewRecognitionResult, ...] | list[PreviewRecognitionResult] = (),
    *,
    error: str = "",
) -> StateMachineResult:
    if error:
        return _result(AssistantState.ERROR, "", False, (error,))

    preview_tuple = tuple(preview_results)
    if preview_tuple:
        accepted = accepted_count(list(preview_tuple))
        low_quality = tuple(result for result in preview_tuple if result.status != "accepted")
        diagnostics = _preview_diagnostics(preview_tuple)
        if accepted > 0:
            if accepted >= 2:
                return _result(
                    AssistantState.ANALYZE_MATCHUP,
                    "team_preview",
                    True,
                    (
                        f"team preview detected: {accepted}/{len(preview_tuple)} opponent slots recognized",
                        *diagnostics,
                    ),
                )
            return _result(
                AssistantState.DETECT_TEAM_PREVIEW,
                "team_preview",
                True,
                (
                    f"team preview partially detected: {accepted}/{len(preview_tuple)} opponent slots recognized",
                    *diagnostics,
                ),
            )
        if len(low_quality) == len(preview_tuple):
            return _result(
                AssistantState.UNKNOWN,
                "team_preview",
                False,
                (
                    "preview slots found, but no confident opponent recognition yet",
                    *diagnostics,
                ),
            )
        return _result(
            AssistantState.UNKNOWN,
            "team_preview",
            False,
            ("preview slots found, but no confident opponent recognition yet", *diagnostics),
        )

    if snapshot is None:
        return _result(AssistantState.UNKNOWN, "", False, ("waiting for screenshot",))

    active_known = len(snapshot.active_pokemon("player")) + len(snapshot.active_pokemon("opponent"))
    if active_known >= 2:
        return _result(
            AssistantState.RECOMMEND,
            "battle_active",
            True,
            (f"battle analysis ready: {active_known} active Pokemon recognized",),
        )
    if active_known > 0:
        return _result(
            AssistantState.DETECT_ACTIVE,
            "battle_active",
            True,
            (f"battle active: {active_known} active Pokemon recognized",),
        )
    if snapshot.turn_text:
        return _result(
            AssistantState.WAIT_BATTLE,
            "battle_active",
            False,
            ("battle UI signal found, waiting for active Pokemon recognition",),
        )
    return _result(AssistantState.UNKNOWN, "", False, ("no reliable battle state recognized",))


def _result(
    state: AssistantState,
    screen: str,
    should_refresh: bool,
    diagnostics: tuple[str, ...],
) -> StateMachineResult:
    return StateMachineResult(
        state=state,
        screen=screen,
        should_refresh_recommendations=should_refresh,
        diagnostics=diagnostics,
        message="; ".join(diagnostics),
    )


def _preview_diagnostics(results: tuple[PreviewRecognitionResult, ...]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    statuses = sorted({result.status for result in results if result.status != "accepted"})
    if statuses:
        diagnostics.append("non-accepted slots: " + ", ".join(statuses))
    failure_reasons = sorted({result.failure_reason for result in results if result.failure_reason})
    diagnostics.extend(failure_reasons)
    close_candidates = [
        result.roi_key
        for result in results
        if result.second_species_id
        and result.thresholds
        and result.confidence - result.second_confidence < result.thresholds.get("ambiguity_margin", 0.0)
    ]
    if close_candidates:
        diagnostics.append("multiple candidates are close: " + ", ".join(close_candidates))
    return tuple(diagnostics)
