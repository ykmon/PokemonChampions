import unittest

from champions_assistant.models import BattleSnapshot, PokemonIdentity, Rect
from champions_assistant.preview_recognition import PreviewRecognitionResult
from champions_assistant.state_machine import AssistantState, evaluate_readonly_state


class StateMachineTests(unittest.TestCase):
    def test_error_state_wins(self):
        result = evaluate_readonly_state(error="bad screenshot")
        self.assertEqual(result.state, AssistantState.ERROR)
        self.assertFalse(result.refresh_recommendations)

    def test_team_preview_when_any_slot_accepted(self):
        result = evaluate_readonly_state(preview_results=[
            _preview_result("accepted"),
            _preview_result("low-confidence"),
        ])
        self.assertEqual(result.state, AssistantState.DETECT_TEAM_PREVIEW)
        self.assertEqual(result.screen, "team_preview")
        self.assertTrue(result.refresh_recommendations)

    def test_analyze_matchup_when_multiple_preview_slots_accepted(self):
        result = evaluate_readonly_state(preview_results=[
            _preview_result("accepted"),
            _preview_result("accepted"),
            _preview_result("low-confidence"),
        ])

        self.assertEqual(result.state, AssistantState.ANALYZE_MATCHUP)
        self.assertTrue(result.should_refresh_recommendations)

    def test_battle_analysis_ready_when_both_active_known(self):
        pikachu = PokemonIdentity(name="Pikachu", species_id="pikachu", types=("electric",))
        gyarados = PokemonIdentity(name="Gyarados", species_id="gyarados", types=("water", "flying"))
        snapshot = BattleSnapshot.from_pair(pikachu, gyarados)

        result = evaluate_readonly_state(snapshot)

        self.assertEqual(result.state, AssistantState.RECOMMEND)
        self.assertEqual(result.screen, "battle_active")
        self.assertTrue(result.refresh_recommendations)

    def test_wait_battle_when_battle_ui_seen_without_active_names(self):
        snapshot = BattleSnapshot.empty()
        snapshot = type(snapshot)(
            battle_format=snapshot.battle_format,
            player_team=snapshot.player_team,
            opponent_team=snapshot.opponent_team,
            player_active=snapshot.player_active,
            opponent_active=snapshot.opponent_active,
            turn_text="Turn 1",
            source_image=snapshot.source_image,
            captured_at=snapshot.captured_at,
        )

        result = evaluate_readonly_state(snapshot)

        self.assertEqual(result.state, AssistantState.WAIT_BATTLE)
        self.assertFalse(result.refresh_recommendations)

    def test_unknown_without_confident_signal(self):
        result = evaluate_readonly_state(preview_results=[_preview_result("rejected")])
        self.assertEqual(result.state, AssistantState.UNKNOWN)
        self.assertFalse(result.refresh_recommendations)
        self.assertTrue(result.diagnostics)


def _preview_result(status):
    return PreviewRecognitionResult(
        slot_index=1,
        roi_key="opponent_preview_1",
        label="Pikachu" if status == "accepted" else "Unknown",
        species_id="pikachu" if status == "accepted" else None,
        confidence=0.99 if status == "accepted" else 0.0,
        status=status,
        template_path=None,
        crop_rect=Rect(1, 2, 3, 4),
        crop_bytes=b"",
    )


if __name__ == "__main__":
    unittest.main()
