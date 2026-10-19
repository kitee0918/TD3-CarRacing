import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.normalize_2025_logs import NEW_FIELDS, normalize_files


class NormalizeHistoricalLogsTest(unittest.TestCase):
    def test_mixed_header_and_restarted_episode_are_distinct(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            old = ["run_idx", "run_name", "epoch", "is_final", "model_path",
                   "episode_reward", "best_reward_so_far", "total_steps", "sigma",
                   "actor_lr", "critic_lr", "gamma", "batch_size", "max_step_per_epoch",
                   "replay_capacity", "sigma_decay_rate", "initial_sigma", "final_sigma",
                   "target_sigma", "min_epoch_before_stop", "brake_rate",
                   "near_start_ratio", "far_end_ratio", "near_weight"]
            first = dict.fromkeys(old, "")
            first.update(run_name="run_a", epoch="100", episode_reward="12.5",
                         best_reward_so_far="20", sigma="0.2", replay_capacity="5000")
            newer = dict.fromkeys(NEW_FIELDS, "")
            newer.update(run_name="run_a", episode="100", seed="42", episode_reward="50",
                         best_reward_so_far="90", sigma="0.1", replay_capacity="65536",
                         env_name="CarRacing-v3")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(old)
                writer.writerow([first[key] for key in old])
                writer.writerow([newer[key] for key in NEW_FIELDS])
            rows = normalize_files([path])
            self.assertEqual([row["session"] for row in rows], [1, 2])
            self.assertEqual(rows[1]["seed"], "42")
            self.assertEqual(rows[1]["best_train_reward_so_far"], "90")


if __name__ == "__main__":
    unittest.main()
