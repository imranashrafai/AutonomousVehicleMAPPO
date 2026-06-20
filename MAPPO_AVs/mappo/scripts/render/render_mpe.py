"""Run a rendered multi-agent MetaDrive episode.

The filename is retained for compatibility with the original MAPPO project,
but this repository uses MetaDrive rather than the MPE environment.
"""

import argparse
import sys
from pathlib import Path


MAPPO_ROOT = Path(__file__).resolve().parents[2]
if str(MAPPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPPO_ROOT))

from eval import run_episode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="X", help="X, O, T, S, C, CXSOT, random")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_episode(
        args.map,
        max(1, args.num_agents),
        args.seed,
        max(1, args.steps),
    )


if __name__ == "__main__":
    main()
