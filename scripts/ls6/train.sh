#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

module load gcc/11.2.0
module load cuda/12.8

config_path="${1:-configs/dense_baseline.toml}"
export TB_SYSTEM="${TB_SYSTEM:-ls6}"
if [[ -z "${DATA_ROOT+x}" && -n "${SCRATCH:-}" ]]; then
    export DATA_ROOT="$SCRATCH/modded-nanogpt-moe"
fi
if [[ -z "${TB_ROOT+x}" && -n "${STOCKYARD:-}" ]]; then
    export TB_ROOT="$STOCKYARD/tensorboard"
fi

uv run --no-sync torchrun \
    --standalone \
    --nproc_per_node=1 \
    --module modded_nanogpt_moe.train \
    --config "$config_path"
