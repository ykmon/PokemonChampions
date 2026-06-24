import unittest

from champions_assistant.damage import DamageCalculator
from champions_assistant.data_loader import DataRepository


class DamageTests(unittest.TestCase):
    def test_stab_super_effective_damage_estimate(self):
        repository = DataRepository()
        attacker = repository.resolve_pokemon("皮卡丘")
        defender = repository.resolve_pokemon("暴鲤龙")
        move = repository.moves_by_name["Thunderbolt"]

        estimate = DamageCalculator(repository).estimate(attacker, defender, move)

        self.assertTrue(estimate.stab)
        self.assertEqual(estimate.type_multiplier, 4)
        self.assertGreater(estimate.damage_max, estimate.damage_min)
        self.assertGreater(estimate.damage_min, 0)
        self.assertGreater(estimate.percent_max, estimate.percent_min)
        self.assertGreater(estimate.percent_min, 0)
        self.assertIn("STAB", estimate.notes)

    def test_type_immunity_returns_zero_damage(self):
        repository = DataRepository()
        attacker = repository.resolve_pokemon("Pikachu")
        defender = repository.resolve_pokemon("Garchomp")
        move = repository.moves_by_name["Thunderbolt"]

        estimate = DamageCalculator(repository).estimate(attacker, defender, move)

        self.assertEqual(estimate.type_multiplier, 0)
        self.assertEqual(estimate.damage_min, 0)
        self.assertEqual(estimate.damage_max, 0)
        self.assertIn("type immunity", estimate.notes)


if __name__ == "__main__":
    unittest.main()
