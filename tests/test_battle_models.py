import unittest

from champions_assistant.data_loader import DataRepository
from champions_assistant.models import BattleFormat, BattleSnapshot, PokemonIdentity, merge_identity


class BattleModelTests(unittest.TestCase):
    def test_singles_snapshot_has_six_team_slots_and_one_active_slot(self):
        snapshot = BattleSnapshot.empty(BattleFormat.SINGLES_63)

        self.assertEqual(len(snapshot.player_team), 6)
        self.assertEqual(len(snapshot.opponent_team), 6)
        self.assertEqual(len(snapshot.player_active), 1)
        self.assertEqual(len(snapshot.opponent_active), 1)
        self.assertEqual(snapshot.battle_format.selected_team_size, 3)

    def test_doubles_snapshot_has_two_active_slots(self):
        snapshot = BattleSnapshot.empty("doubles64")

        self.assertEqual(len(snapshot.player_team), 6)
        self.assertEqual(len(snapshot.opponent_team), 6)
        self.assertEqual(len(snapshot.player_active), 2)
        self.assertEqual(len(snapshot.opponent_active), 2)
        self.assertEqual(snapshot.battle_format.selected_team_size, 4)

    def test_legacy_pair_properties_remain_available(self):
        repository = DataRepository()
        own = repository.resolve_pokemon("Pikachu")
        opponent = repository.resolve_pokemon("Gyarados")

        snapshot = BattleSnapshot.from_pair(own, opponent)

        self.assertEqual(snapshot.self_pokemon.species_id, "pikachu")
        self.assertEqual(snapshot.opponent_pokemon.species_id, "gyarados")
        self.assertEqual(snapshot.player_team[0].pokemon.species_id, "pikachu")
        self.assertEqual(snapshot.opponent_team[0].pokemon.species_id, "gyarados")

    def test_merge_keeps_identified_species_without_battle_data(self):
        existing = PokemonIdentity(source="template")
        recognized = PokemonIdentity(name="妙蛙花", species_id="no0003_pokemon", source="template", confidence=0.9)

        merged = merge_identity(existing, recognized)

        self.assertEqual(merged.species_id, "no0003_pokemon")
        self.assertFalse(merged.is_known)
        self.assertTrue(merged.is_identified)


if __name__ == "__main__":
    unittest.main()
