# Grouped-GEMM backend

By default, the grouped MoE backend uses the external `nv-grouped-gemm` package. It is
intentionally not part of the cross-platform lock file because it must be built
against the target CUDA architecture.

The validated LS6 build used `nv-grouped-gemm==1.1.4.post8`, CUDA 12.8, and an
A100 (SM80). Build it on a GPU compute node, leave `TORCH_CUDA_ARCH_LIST`
unset, and verify that the compiler command includes
`-DGROUPED_GEMM_DEVICE_CAPABILITY=80`.

```bash
unset TORCH_CUDA_ARCH_LIST
GROUPED_GEMM_FORCE_BUILD=TRUE \
uv pip install --no-build-isolation --no-cache -v \
  "nv-grouped-gemm==1.1.4.post8" 2>&1 | tee /tmp/grouped_gemm_build.log
grep -- "-DGROUPED_GEMM_DEVICE_CAPABILITY=80" /tmp/grouped_gemm_build.log
```

No extension rebuild is needed merely to change model configuration. Run the
correctness and benchmark harness from the repository root:

```bash
uv run --no-sync python -m tools.validate_grouped_gemm --help
```

The same package has run on Vista GH200 through its cuBLAS fallback, but the
pinned release's CUTLASS path is specific to SM80. Treat performance and build
behavior independently on each system.
# GH200 fragmentation experiment (PyTorch 2.11)

`MOE_GMM_IMPLEMENTATION=extension` (the default) retains the existing backend.
`MOE_GMM_IMPLEMENTATION=torch` opts into the existing PyTorch SM90 BF16 CUTLASS
grouped kernel. No dependency installation or extension rebuild is needed.
This is an **unmeasured candidate**, not a claimed performance improvement.
It leaves TOMLs, parameters/checkpoint keys, bias/combine, routing and Muon alone.
Keep the same implementation across a resume experiment: the selector is runtime
execution policy, not a new checkpoint compatibility field. Floating-point
reduction order can differ; bitwise cross-backend continuation is not promised.

## Source diagnosis and options

Inspected pinned upstream source, not a locally installed CUDA binary:

- [nv-grouped-gemm ops.py, v1.1.4.post8](https://github.com/fanshiqing/grouped_gemm/blob/15721c6f478092f58ba0767fb0f3642c4f493748/grouped_gemm/ops.py):
  `MoE._gmm` binds `grouped_gemm.ops.gmm`, which applies `GroupedGemm` autograd.
  Forward calls `backend.gmm(A,W,counts,False,False)`. Backward first makes the
  incoming gradient contiguous, then calls `(grad,W,False,True)` for dX and
  `(A,grad,True,False)` for dW, only when those gradients are needed.
- [CUDA dispatch and loops](https://github.com/fanshiqing/grouped_gemm/blob/15721c6f478092f58ba0767fb0f3642c4f493748/csrc/grouped_gemm.cu):
  GH200 uses `CublasGroupedGemm` for forward/dX and
  `CublasGroupedGemmVariableK` for dW. Both loop over experts, call
  `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT)` per expert, and distribute those
  calls over four auxiliary streams with event dependencies. The packed weight
  pointer is advanced per expert, not copied/repacked internally. Each cuBLAS
  call chooses its internal kernel(s); a `64x64` kernel name is a tile size,
  **not evidence that the extension launches once per tile**.
- Its CUTLASS grouped forward is compiled into dispatch only when capability
  is exactly 80 and trans_b=False. There is no CUTLASS dW or SM90 grouped mode,
  no Python algorithm/workspace knob, no exposed split-K control, and no
  selectable persistent/batched alternative for GH200. Setting the SM80 build
  macro on GH200 is not a supported optimization. The extension itself calls
  cuBLAS, not the cuBLASLt API directly; Lt-named kernels can be cuBLAS internal
  choices. Do not attribute every Lt reduction in a full trace to this path
  without launch correlation.
- [PyTorch 2.11 GroupedBlas.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/GroupedBlas.cpp)
  and [GroupMM.cu](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/GroupMM.cu):
  `torch.nn.functional.grouped_mm` already selects a genuine SM90 BF16 grouped
  CUTLASS TMA warp-specialized kernel, with FP32 accumulation, including
  variable-K dW. It constructs group descriptors in one GPU kernel and launches
  a grouped GEMM, not E cuBLAS GEMMs. Kernel initialization may add a constant
  amount of work. Tile/schedule heuristics are internal, not runtime knobs.
  Non-BF16/non-SM90 cases can fall back to a CPU-offset/per-expert loop, so this
  experiment explicitly rejects them rather than silently measuring that path.

Recommendation implemented: reuse PyTorch's already-built grouped forward,
dX and dW, rather than write another Triton GEMM or modify the extension.
`_grouped_gemm.py` uses `A @ W`, `grad @ W.mT`, and grouped `A.mT @ grad`.
The transposes are views. Device int32 cumulative ends are computed once from
existing counts and shared by both FCs/backward. Row-major activations and
column-major `A.mT` for dW keep segment base pointers 16-byte aligned even for
odd counts; there is no padding, truncation or capacity constraint. The existing
CPU count transfer remains unchanged to isolate this GEMM comparison. Empty
expert dW must be zero (covered by pending CUDA tests).

## Geometry and launch expectations

Balanced N=65536, same active width k*H=3072:

| Per-expert GEMM (M,N,K) | E8/K2/H1536 | E64/K8/H384 |
| --- | --- | --- |
| FC1 forward / FC2 dX | 16384,1536,768 | 8192,384,768 |
| FC2 forward / FC1 dX | 16384,768,1536 | 8192,768,384 |
| FC1 dW | 768,1536,16384 | 768,384,8192 |
| FC2 dW | 1536,768,16384 | 384,768,8192 |

E64 has 8x as many per-expert calls, each with 1/8 the FLOPs. dW has 1/4
the output area and half the reduction length. This explains greater launch
granularity/less per-GEMM parallelism and makes split-K plausible, but does not
prove a particular cuBLAS algorithm caused the wall-time gap. Balanced counts
are an analysis model, not an assumption made by the implementation. Matched
FLOPs are not matched memory traffic: E64 has 4x the routed rows and 2x the
total expert-weight elements.

For 192 layer calls / two updates, E64 extension dispatch issues 12,288 cuBLAS
calls per FC **per phase**, hence 24,576 for dX+dW per FC. The reported ~11,190
instances of one nvjet family are consistent with this but cannot be assigned
exactly to dW/dX without the raw trace; other shapes choose other kernel names,
empty groups may do no work, and split-K may add reductions. Do not equate the
reported family count with total GEMM launches.

Native expectation per FC per phase: 192 grouped GEMM launches, 192 descriptor
preparation launches, plus any constant initialization work. Confirm actual
counts in the traces. New `grouped_gemm.fc{1,2}.{forward,dx,dw}` ranges bracket
the calls inside existing moe.* / moe_bw.* ranges. Extension capture reproduces
its inspected autograd calls exactly; normal uncaptured baseline still calls
the original `ops.gmm`. No synchronization is added to training phases.

## Vista validation and measurements

Run from an allocated GH200, with no stale experiment/checkpoint overrides.
No remote runs were performed locally. Keep profiler runs separate from timings.

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TB_ROOT=
unset MOE_GMM_IMPLEMENTATION
set -euo pipefail
uv run --no-sync python -m pytest -q -rs tests/test_native_grouped_gemm.py
uv run --no-sync python -m pytest -q -rs tests

stamp="$(date +%Y%m%d-%H%M%S)-$$"
logs="$STOCKYARD/logs/modded-nanogpt-moe/vista/gemm-$stamp"
profiles="$STOCKYARD/profiles/modded-nanogpt-moe/vista/gemm-$stamp"
mkdir -p "$logs" "$profiles"
# Prints actual installed Python/autograd source and extension binary hashes.
# Add --build-log <existing-build.log> if available; do not rebuild to obtain one.
uv run --no-sync python -m tools.benchmark_expert_gemm --inspect-only \
  2>&1 | tee "$logs/installed-backend.log"

# Isolated actual-size GEMMs: identical seeds/counts/inputs across implementations.
# Device-wide synchronized timing, followed by separate per-phase profiles.
for geometry in e64 e8; do
  for implementation in extension torch; do
    uv run --no-sync python -m tools.benchmark_expert_gemm \
      --geometry "$geometry" --implementation "$implementation" \
      --profile-dir "$profiles/raw-$geometry-$implementation" \
      2>&1 | tee "$logs/raw-$geometry-$implementation.log"
  done
done
# Also use --routing imbalanced for the valid same-k-experts-for-all-tokens case.

unset TRAINING_BENCHMARK NSYS_PROFILE NSYS_WARMUP_STEPS NSYS_ACTIVE_STEPS
unset SEED_OVERRIDE MBS_OVERRIDE TRAIN_STEPS_OVERRIDE MLP_TYPE_OVERRIDE
unset MLP_RATIO_OVERRIDE NUM_EXPERTS_OVERRIDE TOP_K_OVERRIDE MOE_BACKEND_OVERRIDE
unset CHECKPOINT_DIR CHECKPOINT_INTERVAL CHECKPOINT_ROOT CHECKPOINT_POLICY_DISABLED
unset RESUME_CHECKPOINT STOP_AFTER_COMPLETED_UPDATES
unset REPRO_DIAGNOSTICS_DIR BENCHMARK_WARMUP_UPDATES BENCHMARK_MEASURED_UPDATES
for config in configs/moe_e64k8_r0.5.toml configs/moe_grouped.toml; do
  name="$(basename "$config" .toml)"
  for implementation in extension torch; do
    MOE_GMM_IMPLEMENTATION="$implementation" scripts/vista/benchmark.sh "$config" \
      2>&1 | tee "$logs/$name-$implementation.log"
    report="$profiles/$name-$implementation"
    MOE_GMM_IMPLEMENTATION="$implementation" TRAIN_STEPS_OVERRIDE=14 \
    NSYS_PROFILE=1 NSYS_WARMUP_STEPS=10 NSYS_ACTIVE_STEPS=2 \
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop -o "$report" \
      uv run --no-sync torchrun --standalone --nproc_per_node=1 \
      --module modded_nanogpt_moe.train --config "$config" \
      2>&1 | tee "$report.log"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_kern_sum \
      --format csv "$report.nsys-rep" | tee "$report.stats.csv"
    grep -E 'grouped_gemm|nvjet|cutlass|splitK|prepare_grouped' "$report.stats.csv" || true
  done
done
```

Compare forward/dX/dW GPU time and launch counts, plus unprofiled optimizer-update
median and memory. Do not add overlapping operator/kernel/range totals. Phase
timings include host dispatch and output allocation, but exclude upstream casts
and offset construction; full-pipeline benchmarks include them. Sequential
device-wide timing limits auxiliary-stream overlap across calls; full training
is the final judge. Different reduction orders, imbalanced groups, thermal state,
native tile heuristics and GEMM workspace can affect the comparison. Check E8
regression before enabling native execution routinely.
