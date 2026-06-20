import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import setproctitle
import torch
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation

MAPPO_ROOT = Path(__file__).resolve().parents[1]
if str(MAPPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPPO_ROOT))

from config import get_config


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="Intersection_MAPPO", help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--eval", action="store_true", default=False, help="Run evaluation only")
    parser.add_argument("--eval_model_dir", type=str, default=None, help="Directory to load model from for evaluation")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # ---- NEW: force correct recurrent settings per algorithm ----
    if all_args.algorithm_name == "rmappo":
        # RMAPPO MUST be recurrent. If user accidentally turned it off, turn it back on.
        if not (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy):
            all_args.use_recurrent_policy = True

    elif all_args.algorithm_name == "mappo":
        # MAPPO MUST be feed-forward (no recurrence).
        # Force both flags off so the assertion will always pass.
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False

    else:
        raise NotImplementedError

    # (you can completely delete the old assert block)


    assert (
                   all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
           ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_base = (
        MAPPO_ROOT
        / "results"
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_base.mkdir(parents=True, exist_ok=True)
    existing_run_numbers = []
    for folder in run_base.iterdir():
        if not folder.is_dir() or not folder.name.startswith("run"):
            continue
        try:
            existing_run_numbers.append(int(folder.name[3:]))
        except ValueError:
            continue

    curr_run = f"run{max(existing_run_numbers, default=0) + 1}"
    all_args.run_num = curr_run
    run_dir = run_base / curr_run
    run_dir.mkdir(parents=True, exist_ok=False)

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    num_agents = all_args.num_agents

    config_train = dict(
        horizon=5000,
        random_spawn_lane_index=False,
        #start_seed=0,
        use_render=all_args.use_render,
        crash_done=True,
        delay_done=True,
        start_seed=random.randint(0, 1000),
        show_coordinates=True,
        random_traffic=all_args.human_vehicle,
        traffic_density=all_args.traffic_density[all_args.env] if all_args.human_vehicle else 0,
        # agent_policy=IDMPolicy,
        # agent_policy= ExpertPolicy,
        agent_observation=LidarStateObservation,

        
        num_agents=num_agents,
        vehicle_config=dict(
            
            show_navi_mark=all_args.show_navi,
            show_dest_mark=all_args.show_navi,
            show_line_to_dest=all_args.show_navi,
            use_special_color=True,
            random_color=False,
            lidar=dict(
                add_others_navi=False,
                num_others=4,
                distance=50,
                num_lasers=30,
            ),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12),
        )
    )

    config_eval = dict(
        horizon=1000,
        show_fps=True,
        random_spawn_lane_index=False,
        use_render=all_args.use_render_eval,
        crash_done=True,
        delay_done=True,
        allow_respawn=True,
        sensors=dict(rgb_camera=(RGBCamera, 100, 50)),
        start_seed=random.randint(0, 1000),
        out_of_road_done=True,
        interface_panel=["lidar", "dashboard"],
        random_traffic=all_args.human_vehicle,
        traffic_density=all_args.traffic_density[all_args.env] if all_args.human_vehicle else 0,
        # agent_policy=IDMPolicy,
        # agent_policy= ExpertPolicy,
        agent_observation=LidarStateObservation,
        
        num_agents=num_agents,
        vehicle_config=dict(
            
            show_navi_mark=all_args.show_navi,
            show_dest_mark=all_args.show_navi,
            show_line_to_dest=all_args.show_navi,
            use_special_color=True,
            random_color=False,
            lidar=dict(
                add_others_navi=False,
                num_others=4,
                distance=50,
                num_lasers=30,
            ),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12),
        )
    )
    total_reward = []
    config = {
        "all_args": all_args,
        "config_train": config_train,
        "config_eval": config_eval,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "total_reward": total_reward,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    if all_args.eval:
        if all_args.eval_model_dir is None:
            raise ValueError("Model directory must be specified for evaluation")
        
        # Load pre-trained model
        config["model_dir"] = all_args.eval_model_dir

        #Create Runner object
        runner = Runner(config)

        # Run the evaluation
        runner.eval_warmup()
        runner.eval(0)

        # Close the environment
        runner.eval_envs.close()
    else:
        # Normal training process
        runner = Runner(config)
        runner.run()

        # Post-processing
        runner.envs.close()
        if all_args.use_eval:
            runner.eval_envs.close()

        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()

        if config["total_reward"]:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(config["total_reward"]) + 1), config["total_reward"])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Training Reward")
            ax.set_title("Training Reward")
            fig.tight_layout()
            fig.savefig(run_dir / "training_reward.png")
            plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])