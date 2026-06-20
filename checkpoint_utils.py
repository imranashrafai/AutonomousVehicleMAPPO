"""Checkpoint discovery helpers shared by both dashboards."""

from pathlib import Path
from typing import Dict, List


def find_run_folders(algorithm_path: Path) -> List[Path]:
    if not algorithm_path.exists():
        return []
    return sorted(
        (
            run_path
            for experiment in algorithm_path.iterdir()
            if experiment.is_dir()
            for run_path in experiment.glob("run*")
            if run_path.is_dir()
        ),
        key=lambda path: (path.parent.name, path.name),
    )


def find_all_checkpoints(checkpoint_base: Path) -> Dict[str, List[dict]]:
    checkpoints = {"MAPPO": [], "RMAPPO": []}

    for algorithm in ("mappo", "rmappo"):
        for run_folder in find_run_folders(checkpoint_base / algorithm):
            model_path = run_folder / "models"
            shared_policy = (model_path / "actor.pt").is_file()
            separated_policy = (model_path / "actor_agent0.pt").is_file()
            if not (shared_policy or separated_policy):
                continue

            experiment = run_folder.parent.name
            checkpoints[algorithm.upper()].append(
                {
                    "run": run_folder.name,
                    "label": f"{experiment}/{run_folder.name}",
                    "experiment": experiment,
                    "path": str(model_path),
                    "algo": algorithm,
                    "shared_policy": shared_policy,
                    "full_path": str(run_folder),
                }
            )

    return checkpoints
