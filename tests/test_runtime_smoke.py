"""Small integration checks for the published model and environment."""

import importlib.util
import os
from pathlib import Path
import unittest


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "gymnasium", "numpy", "Box2D"))


@unittest.skipUnless(HAS_RUNTIME, "install the project dependencies for runtime checks")
class RuntimeSmokeTests(unittest.TestCase):
    def test_checkpoint_and_environment(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import numpy as np
        import torch

        from td3_carracing.env import FrameStackState, make_env
        from td3_carracing.model import Actor

        checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/course_2025_actor_ep200.pt"
        actor = Actor().eval()
        actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

        env = make_env()
        try:
            observation, _ = env.reset(seed=1001)
            state = FrameStackState().reset(observation)
            self.assertEqual(state.shape, (4, 84, 84))
            with torch.inference_mode():
                action = actor(torch.as_tensor(state))[0].numpy()
            self.assertTrue(np.all(action >= env.action_space.low))
            self.assertTrue(np.all(action <= env.action_space.high))
            next_observation, reward, _, _, _ = env.step(action)
            self.assertEqual(next_observation.shape, (84, 84))
            self.assertTrue(np.isfinite(reward))
        finally:
            env.close()

    def test_delayed_actor_update(self):
        import numpy as np
        import torch

        from td3_carracing.agent import TD3Agent
        from td3_carracing.replay import ReplayBuffer

        torch.set_num_threads(2)
        agent = TD3Agent(torch.device("cpu"))
        replay = ReplayBuffer(2)
        state = np.zeros((4, 84, 84), dtype=np.float32)
        for _ in range(2):
            replay.add(state, np.array([0.0, 0.5, 0.0], dtype=np.float32),
                       state, 0.0, False)
        self.assertNotIn("actor_loss", agent.update(replay, 2))
        self.assertIn("actor_loss", agent.update(replay, 2))


if __name__ == "__main__":
    unittest.main()
