#!/usr/bin/env sh
set -eu

# Compatibility wrapper retained from the original MAPPO layout.
# Environment variables can override the defaults; extra CLI flags are passed
# directly to train/train.py.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MAPPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$MAPPO_ROOT"

ALGORITHM=${ALGORITHM:-rmappo}
ENVIRONMENT=${ENVIRONMENT:-intersection}
EXPERIMENT=${EXPERIMENT:-baseline}
NUM_AGENTS=${NUM_AGENTS:-4}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-100000}
SEED=${SEED:-1}

python train/train.py \
  --algorithm_name "$ALGORITHM" \
  --env "$ENVIRONMENT" \
  --experiment_name "$EXPERIMENT" \
  --num_agents "$NUM_AGENTS" \
  --num_env_steps "$NUM_ENV_STEPS" \
  --seed "$SEED" \
  "$@"
