#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

module load nvidia/25.3
module load cuda/12.9
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

config_path="${1:-configs/dense_baseline.toml}"
export TB_SYSTEM="${TB_SYSTEM:-vista}"
if [[ -z "${DATA_ROOT+x}" ]]; then
    export DATA_ROOT="$repo_root"
fi
if [[ -z "${TB_ROOT+x}" && -n "${STOCKYARD:-}" ]]; then
    export TB_ROOT="$STOCKYARD/tensorboard"
fi

uv run --no-sync torchrun \
    --standalone \
    --nproc_per_node=1 \
    --module modded_nanogpt_moe.train \
    --config "$config_path"
