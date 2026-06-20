"""Compatibility helpers for MetaDrive multi-agent simulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import List

import numpy as np


def active_agent_ids(env) -> List[str]:
    """Return current controllable agent IDs across MetaDrive versions."""
    for attribute in ("agents", "vehicles"):
        agents = getattr(env, attribute, None)
        if isinstance(agents, Mapping):
            return list(agents.keys())
    return []


def sample_multi_agent_actions(env, *, accelerate: bool = False) -> dict:
    """Sample one valid action per active agent.

    MetaDrive versions expose either a shared action space or a Dict-like space.
    The previous dashboard sampled the whole Dict once per agent, producing
    malformed nested actions on newer versions.
    """
    agent_ids = active_agent_ids(env)
    action_space = env.action_space
    spaces = getattr(action_space, "spaces", None)

    if not agent_ids and isinstance(spaces, Mapping):
        agent_ids = list(spaces.keys())

    actions = {}
    for agent_id in agent_ids:
        space = spaces.get(agent_id) if isinstance(spaces, Mapping) else action_space
        if space is None:
            continue
        action = space.sample()

        if accelerate and isinstance(action, np.ndarray) and action.size >= 2:
            action = action.astype(np.float32, copy=True)
            action[0] = 0.0
            action[1] = 1.0

        actions[agent_id] = action
    return actions


def step_environment(env, actions: dict) -> bool:
    """Step an environment and report whether the episode is complete."""
    result = env.step(actions)
    if not isinstance(result, tuple):
        return False

    if len(result) == 5:
        _, _, terminated, truncated, _ = result
        return _all_agents_done(terminated) or _all_agents_done(truncated)
    if len(result) == 4:
        _, _, done, _ = result
        return _all_agents_done(done)
    return False


def select_render_frame(frame):
    """Select a displayable RGB frame from MetaDrive render output."""
    if isinstance(frame, Mapping):
        return frame.get("main_camera", next(iter(frame.values()), None))
    return frame


def _all_agents_done(done) -> bool:
    if isinstance(done, Mapping):
        if "__all__" in done:
            return bool(done["__all__"])
        values = [bool(value) for value in done.values()]
        return bool(values) and all(values)
    if isinstance(done, (list, tuple, np.ndarray)):
        return len(done) > 0 and all(bool(value) for value in done)
    return bool(done)
