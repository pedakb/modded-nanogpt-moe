#!/usr/bin/env bash
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=vista-train-configs

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script_path="$repo_root/scripts/vista/train.sh"
cd "$repo_root"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/vista/train.sh [options] [CONFIG ...]

Run configs on the current allocated node, or submit them as one Slurm suite.
If no config is supplied, configs/moe_e64k8_r0.5.toml is used.

Options:
  --submit                 Submit this script through sbatch
  --time SLURM_TIME        Override Slurm walltime (requires --submit)
  --job-name NAME          Set the Slurm job name (requires --submit)
  --account ACCOUNT        Set the Slurm allocation account (requires --submit)
  --sbatch-arg OPTION      Forward one --option[=value] to sbatch (repeatable)
  --steps N                Stop after N completed updates; preserve configured schedule
  --checkpoint-interval N  Save per-run checkpoints every N updates (0 = final only)
  --resume PATH            Resume the first config from PATH
  -h, --help               Show this help
EOF
}

submit=0
worker=0
benchmark_worker=0
walltime=""
walltime_set=0
sbatch_options=()
steps=""
checkpoint_interval=""
resume_checkpoint=""
config_arguments=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            submit=1
            shift
            ;;
        --time)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            walltime="$2"
            walltime_set=1
            shift 2
            ;;
        --steps)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            steps="$2"
            shift 2
            ;;
        --job-name|--account|--sbatch-arg)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "Error: $1 requires a nonempty value" >&2
                exit 2
            fi
            if [[ "$1" == --sbatch-arg ]]; then
                sbatch_options+=("$2")
            else
                if [[ "$2" == -* ]]; then
                    echo "Error: $1 requires a value, not another option" >&2
                    exit 2
                fi
                sbatch_options+=("$1=$2")
            fi
            shift 2
            ;;
        --sbatch-arg=*)
            sbatch_options+=("${1#*=}")
            shift
            ;;
        --checkpoint-interval)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            checkpoint_interval="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            resume_checkpoint="$2"
            shift 2
            ;;
        --worker)
            worker=1
            shift
            ;;
        --benchmark-worker)
            benchmark_worker=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            config_arguments+=("$@")
            break
            ;;
        --*)
            echo "Error: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            config_arguments+=("$1")
            shift
            ;;
    esac
done

if [[ "$submit" -eq 1 && "$worker" -eq 1 ]]; then
    echo "Error: --submit and internal worker mode cannot be combined" >&2
    exit 2
fi
if [[ "$walltime_set" -eq 1 && "$submit" -ne 1 ]]; then
    echo "Error: --time requires --submit" >&2
    exit 2
fi
if [[ "$walltime_set" -eq 1 && -z "$walltime" ]]; then
    echo "Error: --time requires a nonempty Slurm time value" >&2
    exit 2
fi
if [[ ${#sbatch_options[@]} -gt 0 ]]; then
    if [[ "$submit" -ne 1 ]]; then
        echo "Error: --job-name, --account and --sbatch-arg require --submit" >&2
        exit 2
    fi
    for option in "${sbatch_options[@]}"; do
        if [[ ! "$option" =~ ^--[a-zA-Z][a-zA-Z0-9-]*(=.*)?$ || "$option" == --wrap || "$option" == --wrap=* ]]; then
            echo "Error: expected one --option[=value] for sbatch (no --wrap), got: $option" >&2
            exit 2
        fi
    done
fi
if [[ -n "$steps" && ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --steps must be a positive integer, got: $steps" >&2
    exit 2
fi
if [[ -n "$checkpoint_interval" && ! "$checkpoint_interval" =~ ^[0-9]+$ ]]; then
    echo "Error: --checkpoint-interval must be a nonnegative integer" >&2
    exit 2
fi
if [[ "$submit" -eq 1 && -n "$steps" ]]; then
    echo "Error: --steps is for current-node smoke tests and cannot be used with --submit" >&2
    exit 2
fi
if [[ -n "$steps" && ( -n "$checkpoint_interval" || -n "$resume_checkpoint" ) ]]; then
    echo "Error: --steps cannot be combined with checkpoint or resume options" >&2
    exit 2
fi
if [[ "$benchmark_worker" -eq 1 && ( "$submit" -eq 1 || "$worker" -eq 1 ) ]]; then
    echo "Error: benchmark worker mode cannot submit or run as a Slurm worker" >&2
    exit 2
fi
if [[ "$benchmark_worker" -eq 1 && ( -n "$steps" || -n "$checkpoint_interval" || -n "$resume_checkpoint" ) ]]; then
    echo "Error: benchmark worker mode cannot use step, checkpoint, or resume options" >&2
    exit 2
fi

if [[ ${#config_arguments[@]} -eq 0 ]]; then
    config_arguments=(configs/moe_e64k8_r0.5.toml)
fi
if [[ -n "$steps" && ${#config_arguments[@]} -ne 1 ]]; then
    echo "Error: --steps requires exactly one config" >&2
    exit 2
fi
if [[ "$benchmark_worker" -eq 1 && ${#config_arguments[@]} -ne 1 ]]; then
    echo "Error: benchmark worker mode requires exactly one config" >&2
    exit 2
fi

# Resolve every config and its TOML run identity before submission or training.
config_paths=()
run_names=()
declare -A seen_run_names=()
for config_argument in "${config_arguments[@]}"; do
    config_path="$config_argument"
    if [[ "$config_path" != /* ]]; then
        config_path="$repo_root/$config_path"
    fi
    if [[ ! -f "$config_path" ]]; then
        echo "Error: config file not found: $config_argument" >&2
        exit 2
    fi

    run_name="$(sed -nE \
        -e 's/^[[:space:]]*run_name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
        -e "s/^[[:space:]]*run_name[[:space:]]*=[[:space:]]*'([^']+)'.*/\1/p" \
        "$config_path")"
    if [[ -z "$run_name" || "$run_name" == *$'\n'* ]]; then
        echo "Error: expected exactly one quoted run_name in $config_argument" >&2
        exit 2
    fi
    if [[ "$run_name" == */* || "$run_name" == *\\* || "$run_name" == "." || "$run_name" == ".." ]]; then
        echo "Error: run_name must be a single directory-safe name: $run_name" >&2
        exit 2
    fi
    if [[ -n "${seen_run_names[$run_name]:-}" ]]; then
        echo "Error: duplicate run_name in suite: $run_name" >&2
        exit 2
    fi
    seen_run_names["$run_name"]=1
    config_paths+=("$config_path")
    run_names+=("$run_name")
done

if [[ -n "$resume_checkpoint" ]]; then
    if [[ "$resume_checkpoint" != /* ]]; then
        resume_checkpoint="$repo_root/$resume_checkpoint"
    fi
    if [[ ! -f "$resume_checkpoint" ]]; then
        echo "Error: resume checkpoint not found: $resume_checkpoint" >&2
        exit 2
    fi
fi

if [[ "$submit" -eq 1 ]]; then
    submission=(sbatch)
    if [[ "$walltime_set" -eq 1 ]]; then
        submission+=("--time=$walltime")
    fi
    if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
        submission+=("--mail-user=$SLURM_MAIL_USER" --mail-type=ALL)
    fi
    if [[ ${#sbatch_options[@]} -gt 0 ]]; then
        submission+=("${sbatch_options[@]}")
    fi
    submission+=("$script_path" --worker)
    if [[ -n "$checkpoint_interval" ]]; then
        submission+=(--checkpoint-interval "$checkpoint_interval")
    fi
    if [[ -n "$resume_checkpoint" ]]; then
        submission+=(--resume "$resume_checkpoint")
    fi
    submission+=(-- "${config_paths[@]}")

    printf 'Submitting:'
    printf ' %q' "${submission[@]}"
    printf '\n'
    exec "${submission[@]}"
fi

if [[ "$worker" -eq 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: internal worker mode requires a Slurm allocation" >&2
    exit 2
fi

source "$repo_root/scripts/vista/env.sh"

smoke_checkpoint_dir=""
if [[ -n "$steps" ]]; then
    smoke_checkpoint_dir="$STOCKYARD/checkpoints/modded-nanogpt-moe"
    smoke_checkpoint_dir+="/interactive-smoke/${run_names[0]}"
    smoke_checkpoint_dir+="/$(date '+%Y%m%d-%H%M%S')-$$"
fi

if [[ "$worker" -eq 1 ]]; then
    log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm"
    mkdir -p "$log_dir"
    log_path="$log_dir/train-configs-${SLURM_JOB_ID}.log"
    exec > >(tee -a "$log_path") 2>&1
    echo "Vista suite log: $log_path"
fi

echo "Configs (${#config_paths[@]}), in execution order:"
for index in "${!config_paths[@]}"; do
    printf '  %d. %s (%s)\n' "$((index + 1))" "${config_paths[$index]}" "${run_names[$index]}"
done

# Only command-line options above may establish run-specific state.
benchmark_warmup_updates="${BENCHMARK_WARMUP_UPDATES:-}"
benchmark_measured_updates="${BENCHMARK_MEASURED_UPDATES:-}"
unset SEED_OVERRIDE MBS_OVERRIDE MLP_TYPE_OVERRIDE MLP_RATIO_OVERRIDE
unset NUM_EXPERTS_OVERRIDE TOP_K_OVERRIDE MOE_BACKEND_OVERRIDE
unset TRAIN_STEPS_OVERRIDE TRAINING_BENCHMARK
unset BENCHMARK_WARMUP_UPDATES BENCHMARK_MEASURED_UPDATES
unset NSYS_PROFILE NSYS_WARMUP_STEPS NSYS_ACTIVE_STEPS
unset CHECKPOINT_DIR CHECKPOINT_INTERVAL CHECKPOINT_ROOT
unset CHECKPOINT_POLICY_DISABLED RESUME_CHECKPOINT
unset STOP_AFTER_COMPLETED_UPDATES REPRO_DIAGNOSTICS_DIR
unset MOE_GMM_IMPLEMENTATION

for index in "${!config_paths[@]}"; do
    config_path="${config_paths[$index]}"
    run_name="${run_names[$index]}"
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "==============================================================================="
    echo "START  [$timestamp] run_name=$run_name config=$config_path"
    echo "==============================================================================="

    training_environment=(
        env
        "DATA_ROOT=$repo_root"
        "CHECKPOINT_ROOT=$STOCKYARD/checkpoints/modded-nanogpt-moe"
        MOE_GMM_IMPLEMENTATION=torch
    )
    if [[ -n "$steps" ]]; then
        training_environment+=(
            "STOP_AFTER_COMPLETED_UPDATES=$steps"
            "CHECKPOINT_DIR=$smoke_checkpoint_dir"
            CHECKPOINT_POLICY_DISABLED=1
            "TB_ROOT="
        )
    fi
    if [[ "$benchmark_worker" -eq 1 ]]; then
        training_environment+=(TRAINING_BENCHMARK=1 "TB_ROOT=")
        if [[ -n "$benchmark_warmup_updates" ]]; then
            training_environment+=("BENCHMARK_WARMUP_UPDATES=$benchmark_warmup_updates")
        fi
        if [[ -n "$benchmark_measured_updates" ]]; then
            training_environment+=("BENCHMARK_MEASURED_UPDATES=$benchmark_measured_updates")
        fi
    fi
    if [[ -n "$checkpoint_interval" ]]; then
        training_environment+=("CHECKPOINT_INTERVAL=$checkpoint_interval")
    fi
    if [[ "$index" -eq 0 && -n "$resume_checkpoint" ]]; then
        training_environment+=("RESUME_CHECKPOINT=$resume_checkpoint")
    fi

    if "${training_environment[@]}" \
        uv run --no-sync torchrun \
        --standalone \
        --nproc_per_node=1 \
        --module modded_nanogpt_moe.train \
        --config "$config_path"; then
        timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "COMPLETE [$timestamp] run_name=$run_name config=$config_path"
    else
        status=$?
        timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "FAILED [$timestamp] status=$status run_name=$run_name config=$config_path" >&2
        exit "$status"
    fi
done

echo "Suite complete: $(date '+%Y-%m-%dT%H:%M:%S%z')"
