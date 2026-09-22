#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "Usage: scripts/vista/benchmark.sh <config-path>" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="$1"
if [[ "$config_path" != /* ]]; then
    if [[ -f "$config_path" ]]; then
        config_path="$(cd "$(dirname "$config_path")" && pwd)/$(basename "$config_path")"
    else
        config_path="$repo_root/$config_path"
    fi
fi
if [[ ! -f "$config_path" ]]; then
    echo "Error: config file not found: $1" >&2
    exit 2
fi

for variable in \
    SEED_OVERRIDE MBS_OVERRIDE TRAIN_STEPS_OVERRIDE MLP_TYPE_OVERRIDE \
    MLP_RATIO_OVERRIDE NUM_EXPERTS_OVERRIDE TOP_K_OVERRIDE MOE_BACKEND_OVERRIDE \
    CHECKPOINT_DIR RESUME_CHECKPOINT STOP_AFTER_COMPLETED_UPDATES \
    REPRO_DIAGNOSTICS_DIR; do
    if [[ -n "${!variable:-}" ]]; then
        echo "Error: $variable must be unset for a controlled training benchmark" >&2
        exit 2
    fi
done
if [[ -n "${CHECKPOINT_INTERVAL:-}" && "${CHECKPOINT_INTERVAL}" != 0 ]]; then
    echo "Error: CHECKPOINT_INTERVAL must be unset or 0 for a training benchmark" >&2
    exit 2
fi
if [[ -n "${NSYS_PROFILE:-}" && "${NSYS_PROFILE}" != 0 ]]; then
    echo "Error: NSYS_PROFILE must be unset or 0 for a training benchmark" >&2
    exit 2
fi

cd "$repo_root"
exec "$repo_root/scripts/vista/train.sh" --benchmark-worker "$config_path"
