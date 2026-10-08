#!/usr/bin/env bash
# Current-node only. No production files, jobs, or training settings are changed.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source scripts/vista/env.sh
: "${STOCKYARD:?STOCKYARD must point to persistent storage}"
command -v nsys >/dev/null
command -v uv >/dev/null
configs=(configs/baselines/moe_e64k8.toml configs/moe_architectures/moe_e256k6.toml configs/moe_architectures/moe_e256k6_shared.toml)
reuse_e64=
if [[ $# -gt 0 ]]; then
    if [[ $# -eq 2 && "$1" == --reuse-e64 && -d "$2" ]]; then
        reuse_e64="$(cd "$2" && pwd)"
        # Reuse requires identical scientific source, configs and environment
        # specification, plus saved raw report, log and actual profiler status.
        diff -qr --exclude=__pycache__ modded_nanogpt_moe "$reuse_e64/source/modded_nanogpt_moe"
        for file in pyproject.toml uv.lock; do cmp "$file" "$reuse_e64/source/$file"; done
        for config in "${configs[@]}"; do cmp "$config" "$reuse_e64/configs/$(basename "$config")"; done
        for file in profile.nsys-rep execution.log profile.exit-status; do
            if [[ ! -s "$reuse_e64/moe_e64k8/$file" ]]; then
                echo "Error: missing reuse artifact: $reuse_e64/moe_e64k8/$file" >&2
                exit 1
            fi
        done
    else
        echo 'Usage: bash scripts/vista/profile_experts.sh [--reuse-e64 PREVIOUS_RESULTS_DIR]' >&2
        exit 2
    fi
fi
# Reject scientific overrides rather than silently accepting changed experiments.
for variable in SEED_OVERRIDE MBS_OVERRIDE TRAIN_STEPS_OVERRIDE MLP_TYPE_OVERRIDE MLP_RATIO_OVERRIDE NUM_EXPERTS_OVERRIDE TOP_K_OVERRIDE MOE_BACKEND_OVERRIDE; do
    if [[ -n "${!variable:-}" ]]; then
        echo "Error: unset $variable before profiling production settings" >&2
        exit 2
    fi
done
unset CHECKPOINT_DIR CHECKPOINT_INTERVAL CHECKPOINT_ROOT RESUME_CHECKPOINT
unset STOP_AFTER_COMPLETED_UPDATES
unset REPRO_DIAGNOSTICS_DIR TRAINING_BENCHMARK BENCHMARK_WARMUP_UPDATES BENCHMARK_MEASURED_UPDATES
export TB_ROOT= CHECKPOINT_POLICY_DISABLED=1
export NSYS_PROFILE=1 NSYS_WARMUP_STEPS=5 NSYS_ACTIVE_STEPS=2
export MOE_GMM_IMPLEMENTATION=torch
export DATA_ROOT="${DATA_ROOT:-$repo_root}"
parent="$STOCKYARD/profiles/modded-nanogpt-moe/vista"
mkdir -p "$parent"
results="$(mktemp -d "$parent/expert-scaling-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
echo "Results: $results"
mkdir "$results/source" "$results/configs"
mkdir "$results/runtime-tmp"
export TMPDIR="$results/runtime-tmp"
export UV_CACHE_DIR="$results/uv-cache"
# Snapshot only code/configs: never copy datasets, environments, or artifacts.
cp -a modded_nanogpt_moe "$results/source/"
cp pyproject.toml uv.lock "$results/source/"
cp tools/compare_nsys_experts.py "$results/"
cp scripts/vista/profile_experts.sh "$results/"
for config in "${configs[@]}"; do cp "$config" "$results/configs/"; done
uv run --no-sync python -c 'import sys; from modded_nanogpt_moe.config import load_experiment_config; [load_experiment_config(p) for p in sys.argv[1:]]' "${configs[@]}"
{
    git rev-parse HEAD
    git branch --show-current
    git status --short
    nsys --version
    uv run --no-sync python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda); print(torch.cuda.get_device_name()); print("capability", torch.cuda.get_device_capability())'
    nvidia-smi
    env | sort | grep -E '^(NSYS_|STOP_AFTER_|MOE_|DATA_ROOT|TB_|CHECKPOINT_|CUDA_|TORCH_|CC=|CXX=|SLURM_JOB_ID=)'
} > "$results/environment.txt" 2>&1
git diff --binary > "$results/source.diff"
find "$results/source" "$results/configs" -type f ! -path '*/__pycache__/*' -exec sha256sum {} + > "$results/sha256.txt"
for config in "${configs[@]}"; do
    name="$(basename "$config" .toml)"
    mkdir "$results/$name"
    output="$results/$name/profile"
    export TORCHINDUCTOR_CACHE_DIR="$results/$name/inductor-cache"
    export TRITON_CACHE_DIR="$results/$name/triton-cache"
    if [[ "$name" == moe_e64k8 && -n "$reuse_e64" ]]; then
        echo "Reusing E64 capture: $reuse_e64"
        for file in profile.nsys-rep execution.log profile.exit-status; do
            cp "$reuse_e64/$name/$file" "$results/$name/"
        done
        printf '%s\n' "$reuse_e64" > "$results/$name/reused-from.txt"
        profile_status="$(cat "$results/$name/profile.exit-status")"
    else
    # Reuse the provisioned environment; import the frozen source first.
    # Nsight's intentional SIGTERM can make torchrun/nsys return nonzero.
    # Save both pipeline statuses before restoring fail-fast behavior.
    set +e
    (cd "$results/source"
    PYTHONPATH="$results/source" nsys profile \
        --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=sigterm \
        --force-overwrite=false --output="$output" \
        uv run --project "$repo_root" --no-sync python -m torch.distributed.run --standalone --nproc_per_node=1 \
        --module modded_nanogpt_moe.train --config "$results/configs/$(basename "$config")" \
        ) 2>&1 | tee "$results/$name/execution.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        echo "Error: execution log write failed for $name" >&2
        exit "${pipeline_status[1]}"
    fi
    printf '%s\n' "${pipeline_status[0]}" > "$results/$name/profile.exit-status"
    profile_status="${pipeline_status[0]}"
    fi
    if [[ ! -s "$output.nsys-rep" ]]; then
        echo "Error: Nsight report missing or empty: $output.nsys-rep (see $results/$name/execution.log)" >&2
        exit 1
    fi
    # Export also proves Nsight can open the report. Validate capture BEFORE stats
    # or continuing, even when the profiler command returned zero.
    if ! nsys export --type sqlite --output "$output.sqlite" "$output.nsys-rep" \
        > "$results/$name/export.log" 2>&1; then
        cat "$results/$name/export.log" >&2
        echo "Error: Nsight report export failed for $name" >&2
        exit 1
    fi
    uv run --no-sync python "$results/compare_nsys_experts.py" "$results" \
        --validate-capture "$output.sqlite" --execution-log "$results/$name/execution.log" \
        --profile-status "$profile_status" \
        > "$results/$name/capture-validation.txt" 2>&1 || {
            cat "$results/$name/capture-validation.txt" >&2
            echo "Error: capture validation failed for $name; see execution.log and export.log" >&2
            exit 1
        }
    for report in cuda_gpu_kern_sum cuda_api_sum nvtx_sum nvtx_kern_sum; do
        # Analyze the explicitly exported/validated database directly. Passing
        # .nsys-rep makes stats perform another version-sensitive export check.
        nsys stats --quiet --report "$report" --format csv "$output.sqlite" \
            > "$results/$name/$report.csv" 2> "$results/$name/$report.log"
    done
done
uv run --no-sync python "$results/compare_nsys_experts.py" "$results"
echo "Completed: $results/summary.md"
