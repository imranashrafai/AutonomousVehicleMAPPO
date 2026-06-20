"""Record a portable multi-agent MetaDrive demonstration video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_utils import (
    sample_multi_agent_actions,
    select_render_frame,
    step_environment,
)

def record_video(
    output: Path,
    *,
    map_code: str,
    num_agents: int,
    steps: int,
    fps: int,
    seed: int,
) -> Path:
    try:
        import cv2
        try:
            from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
        except ImportError:
            from metadrive.envs.marl_envs.multi_agent_metadrive import (
                MultiAgentMetaDriveEnv as MultiAgentMetaDrive,
            )
    except ImportError as exc:
        raise RuntimeError(
            "Video recording requires MetaDrive and opencv-python. "
            "Install MAPPO_AVs/requirements.txt first."
        ) from exc

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env = MultiAgentMetaDrive(
        {
            "map": map_code,
            "num_agents": num_agents,
            "start_seed": seed,
            "horizon": max(steps, 1000),
            "use_render": False,
            "offscreen_render": True,
        }
    )
    writer = None

    try:
        env.reset()
        for _ in range(steps):
            actions = sample_multi_agent_actions(env)
            episode_done = step_environment(env, actions)
            frame = select_render_frame(env.render(mode="rgb_array"))

            if isinstance(frame, np.ndarray):
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if episode_done:
                env.reset()
    finally:
        if writer is not None:
            writer.release()
        env.close()

    if writer is None:
        raise RuntimeError("MetaDrive did not return any RGB frames.")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("videos/metadrive_demo.mp4"))
    parser.add_argument("--map", dest="map_code", default="X")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = record_video(
        args.output,
        map_code=args.map_code,
        num_agents=max(1, args.num_agents),
        steps=max(1, args.steps),
        fps=max(1, args.fps),
        seed=args.seed,
    )
    print(f"Video saved to: {path}")


if __name__ == "__main__":
    main()
