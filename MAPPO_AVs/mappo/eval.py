import argparse
import numpy as np

# Detect correct env class for your MetaDrive 0.4.2
try:
    from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
    ENV_CLS = MultiAgentMetaDrive
except Exception:
    from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDriveEnv
    ENV_CLS = MultiAgentMetaDriveEnv


def make_env(map_code: str, num_agents: int, seed: int, horizon: int = 1000):
    """Create multi-agent MetaDrive environment."""
    config = dict(
        map=map_code,            # X, O, T, S, C, CXSOT, random
        num_agents=num_agents,
        start_seed=seed,
        use_render=True,        # IMPORTANT: show window so you can record it
        horizon=horizon,
        traffic_density=0.2,    # add background traffic so it looks busy
        traffic_mode="hybrid",
    )
    return ENV_CLS(config)


def step_env(env, actions):
    """
    Handle MetaDrive 0.4.x step API (5 returns) and older (4 returns).
    Always returns: obs, rewards, dones, truncations, infos
    """
    try:
        obs, rewards, dones, truncations, infos = env.step(actions)
    except ValueError:
        # Older signature: obs, rewards, dones, infos
        obs, rewards, dones, infos = env.step(actions)
        truncations = {aid: False for aid in dones} if isinstance(dones, dict) else False
    return obs, rewards, dones, truncations, infos


def run_episode(map_code: str, num_agents: int, seed: int, max_steps: int):
    """
    Run one long episode with random actions for ALL env.vehicles.
    This is just for VIDEO: continuous movement, resets when done.
    """
    env = make_env(map_code, num_agents, seed, max_steps)
    obs = env.reset()
    step = 0

    print("[INFO] Env reset. Controllable agents at start:",
          list(getattr(env, "vehicles", {}).keys()))

    try:
        while step < max_steps:
            # Get current controllable agent IDs (this is what your project uses)
            vehicle_ids = list(getattr(env, "vehicles", {}).keys())

            if not vehicle_ids:
                # No controllable vehicles yet, step with empty dict
                obs, rewards, dones, truncations, infos = step_env(env, {})
                env.render()
                step += 1
                continue

            actions = {}

            # Action space is usually a Dict keyed by agent_id
            if hasattr(env.action_space, "spaces"):
                space_dict = env.action_space.spaces
                for agent_id in vehicle_ids:
                    if agent_id in space_dict:
                        actions[agent_id] = space_dict[agent_id].sample()
                    else:
                        # fallback: sample shared space
                        actions[agent_id] = env.action_space.sample()
            else:
                # Shared Box space
                act = env.action_space.sample()
                for agent_id in vehicle_ids:
                    actions[agent_id] = act

            # Do one step
            obs, rewards, dones, truncations, infos = step_env(env, actions)

            # Render frame (window updates)
            env.render()

            # If all agents done → reset env to keep video running
            if isinstance(dones, dict) and len(dones) > 0 and all(dones.values()):
                print(f"[INFO] All agents done at step {step}. Resetting env for continuous video...")
                obs = env.reset()

            step += 1

        print("[INFO] Finished running for", max_steps, "steps.")

    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="X",
                        help="X, O, T, S, C, CXSOT, random")
    parser.add_argument("--num_agents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Running MetaDrive | map={args.map}, agents={args.num_agents}, steps={args.steps}")
    run_episode(args.map, args.num_agents, args.seed, args.steps)
