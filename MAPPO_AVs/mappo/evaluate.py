"""Convenience entry point for evaluating a trained MAPPO/RMAPPO model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional


MAPPO_ROOT = Path(__file__).resolve().parent
if str(MAPPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPPO_ROOT))

from train.train import main as run_experiment


def _evaluation_args(argv: List[str]) -> List[str]:
    args = list(argv)

    # Keep compatibility with the older evaluate.py interface.
    if "--model_dir" in args and "--eval_model_dir" not in args:
        args[args.index("--model_dir")] = "--eval_model_dir"

    if "--eval" not in args:
        args.append("--eval")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    run_experiment(_evaluation_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
