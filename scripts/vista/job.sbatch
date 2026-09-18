#!/usr/bin/env bash
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00

set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "Usage: sbatch scripts/vista/job.sbatch <config-path>" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec "$repo_root/scripts/vista/train.sh" "$1"
