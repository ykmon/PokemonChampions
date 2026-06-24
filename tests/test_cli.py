import io
import unittest
from contextlib import redirect_stdout

from champions_assistant.cli import main


class CliTests(unittest.TestCase):
    def test_legacy_analyze_arguments_still_work(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["analyze", "--self", "Pikachu", "--opponent", "Gyarados", "--move", "Thunderbolt"])

        self.assertEqual(code, 0)
        self.assertIn("Thunderbolt", out.getvalue())

    def test_doubles_analyze_accepts_team_and_active_arguments(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([
                "analyze",
                "--format",
                "doubles64",
                "--self-team",
                "Pikachu,Charizard,Gengar,Lucario,Dragonite,Sylveon",
                "--opponent-team",
                "Gyarados,Venusaur,Garchomp,Metagross,Incineroar,Flutter Mane",
                "--self-active",
                "Pikachu,Charizard",
                "--opponent-active",
                "Gyarados,Venusaur",
            ])

        self.assertEqual(code, 0)
        self.assertIn("己方场上1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
