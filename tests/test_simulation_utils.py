import unittest

import numpy as np

from simulation_utils import (
    active_agent_ids,
    sample_multi_agent_actions,
    select_render_frame,
    step_environment,
)


class _Space:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def sample(self):
        return self.action.copy()


class _DictSpace:
    def __init__(self):
        self.spaces = {
            "agent0": _Space([0.5, -0.5]),
            "agent1": _Space([-0.25, 0.25]),
        }


class _Environment:
    def __init__(self, result=None):
        self.agents = {"agent0": object(), "agent1": object()}
        self.action_space = _DictSpace()
        self.result = result

    def step(self, _actions):
        return self.result


class SimulationUtilsTests(unittest.TestCase):
    def test_samples_each_dict_action_space(self):
        env = _Environment()
        actions = sample_multi_agent_actions(env)
        self.assertEqual(set(actions), {"agent0", "agent1"})
        np.testing.assert_array_equal(actions["agent0"], [0.5, -0.5])

    def test_acceleration_override_does_not_mutate_space_sample(self):
        env = _Environment()
        actions = sample_multi_agent_actions(env, accelerate=True)
        np.testing.assert_array_equal(actions["agent0"], [0.0, 1.0])

    def test_handles_five_value_step_api(self):
        env = _Environment(
            (
                {},
                {},
                {"agent0": True, "agent1": True, "__all__": True},
                {"agent0": False, "agent1": False, "__all__": False},
                {},
            )
        )
        self.assertTrue(step_environment(env, {}))

    def test_helpers_handle_mapping_outputs(self):
        env = _Environment()
        self.assertEqual(active_agent_ids(env), ["agent0", "agent1"])
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertIs(select_render_frame({"main_camera": frame}), frame)


if __name__ == "__main__":
    unittest.main()
