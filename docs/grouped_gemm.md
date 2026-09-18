# Grouped-GEMM backend

The grouped MoE backend uses the external `nv-grouped-gemm` package. It is
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
