#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "Usage: scripts/vista/submit.sh <config-path>" >&2
    exit 2
fi
if [[ -z "${STOCKYARD:-}" ]]; then
    echo "Error: STOCKYARD must be set" >&2
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

log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm"
mkdir -p "$log_dir"
job_name="$(basename "${config_path%.*}")"
mail_args=()
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
    mail_args+=(--mail-user="$SLURM_MAIL_USER" --mail-type=ALL)
fi

cd "$repo_root"
exec sbatch \
    --job-name="$job_name" \
    --output="$log_dir/%x-%j.log" \
    --error="$log_dir/%x-%j.log" \
    "${mail_args[@]}" \
    "$repo_root/scripts/vista/job.sbatch" \
    "$config_path"
