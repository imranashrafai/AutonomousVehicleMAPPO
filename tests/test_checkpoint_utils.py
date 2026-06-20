import tempfile
import unittest
from pathlib import Path

from checkpoint_utils import find_all_checkpoints, find_run_folders


class CheckpointUtilsTests(unittest.TestCase):
    def test_discovers_all_experiments_and_policy_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shared_models = base / "mappo" / "baseline" / "run1" / "models"
            separated_models = base / "rmappo" / "recurrent" / "run2" / "models"
            shared_models.mkdir(parents=True)
            separated_models.mkdir(parents=True)
            (shared_models / "actor.pt").touch()
            (separated_models / "actor_agent0.pt").touch()

            checkpoints = find_all_checkpoints(base)

            self.assertEqual(checkpoints["MAPPO"][0]["label"], "baseline/run1")
            self.assertTrue(checkpoints["MAPPO"][0]["shared_policy"])
            self.assertEqual(checkpoints["RMAPPO"][0]["label"], "recurrent/run2")
            self.assertFalse(checkpoints["RMAPPO"][0]["shared_policy"])

    def test_ignores_runs_without_actor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithm_path = Path(directory) / "mappo"
            (algorithm_path / "experiment" / "run1" / "models").mkdir(parents=True)

            self.assertEqual(find_run_folders(algorithm_path)[0].name, "run1")
            self.assertEqual(
                find_all_checkpoints(Path(directory))["MAPPO"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
