import unittest

from champions_assistant.data_loader import DataRepository


class TypeChartTests(unittest.TestCase):
    def test_type_chart_core_matchups(self):
        repository = DataRepository()
        chart = repository.type_chart

        self.assertEqual(chart.multiplier("Electric", ("Water", "Flying")), 4)
        self.assertEqual(chart.multiplier("Ground", ("Flying",)), 0)
        self.assertEqual(chart.multiplier("Normal", ("Ghost",)), 0)
        self.assertEqual(chart.multiplier("Fire", ("Steel",)), 2)
        self.assertEqual(chart.multiplier("Fairy", ("Dragon",)), 2)

    def test_defender_profile_groups_weak_resist_and_immune_types(self):
        repository = DataRepository()
        profile = repository.type_chart.defender_profile(("Ghost", "Fairy"))

        self.assertIn("Ghost", profile.weak_to)
        self.assertIn("Dragon", profile.immune_to)
        self.assertIn("Bug", profile.resists)


if __name__ == "__main__":
    unittest.main()
