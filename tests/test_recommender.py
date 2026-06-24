import unittest

from champions_assistant.cli import _snapshot_from_cli
from champions_assistant.data_loader import DataRepository
from champions_assistant.recommender import build_recommendations


class RecommenderTests(unittest.TestCase):
    def test_singles_recommendations_include_current_matchup(self):
        repository = DataRepository()
        snapshot = _snapshot_from_cli(
            repository,
            battle_format="singles63",
            self_name="Pikachu",
            opponent_name="Gyarados",
            self_team="",
            opponent_team="",
            self_active="",
            opponent_active="",
        )

        recommendations = build_recommendations(snapshot, repository)
        reasons = "\n".join(item.reason for item in recommendations)

        self.assertIn("Pikachu", reasons)
        self.assertIn("Gyarados", reasons)
        self.assertTrue(any(item.title == "进攻机会" for item in recommendations))

    def test_doubles_recommendations_label_slots(self):
        repository = DataRepository()
        snapshot = _snapshot_from_cli(
            repository,
            battle_format="doubles64",
            self_name=None,
            opponent_name=None,
            self_team="Pikachu,Charizard,Gengar,Lucario,Dragonite,Sylveon",
            opponent_team="Gyarados,Venusaur,Garchomp,Metagross,Incineroar,Flutter Mane",
            self_active="Pikachu,Charizard",
            opponent_active="Gyarados,Venusaur",
        )

        recommendations = build_recommendations(snapshot, repository)
        text = "\n".join(item.reason for item in recommendations)

        self.assertIn("己方场上1", text)
        self.assertIn("对方场上1", text)
        self.assertTrue(any(item.title == "伤害样本" for item in recommendations))


if __name__ == "__main__":
    unittest.main()
