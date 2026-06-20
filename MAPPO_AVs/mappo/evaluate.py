# evaluate_env.py
import sys
import os
import torch
from pathlib import Path
from config import get_config  # Your existing config parser

# Add your MetaDrive site-packages if needed
VENV_SITE_PACKAGES_PATH = r"D:\AutonomousVehicleFYP\marl_fyp\Lib\site-packages"
if VENV_SITE_PACKAGES_PATH not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES_PATH)

# Runner imports
parser = get_config()
all_args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

if all_args.share_policy:
    from runner.shared.env_runner import EnvRunner as Runner
else:
    from runner.separated.env_runner import EnvRunner as Runner

def main():
    if all_args.model_dir is None:
        raise ValueError("Please provide --model_dir to load the pretrained model.")

    # Device setup
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Build config dict for Runner
    config = {
        "all_args": all_args,
        "device": device,
        "num_agents": all_args.num_agents,
        "config_eval": {
            "horizon": 1000,
            "use_render": all_args.use_render_eval,
            "crash_done": True,
            "delay_done": True,
            "allow_respawn": True,
            "num_agents": all_args.num_agents,
        },
        "model_dir": all_args.model_dir,
    }

    # Initialize Runner
    runner = Runner(config)

    print(f"Evaluating model from: {all_args.model_dir}")
    runner.eval_warmup()  # Warm-up if required
    runner.eval(0)         # Run evaluation for 1 iteration

    print("Evaluation finished.")
    runner.eval_envs.close()

if __name__ == "__main__":
    main()
