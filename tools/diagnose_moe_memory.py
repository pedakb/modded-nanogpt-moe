"""
Standalone LS6 memory/compile diagnostic for the MoE(E=1,k=1) OOM.

Imports the active model package without editing it. Does not change
router/optimizer semantics, does not add an E=1 shortcut to the model code,
does not implement padded-bmm or grouped GEMM, does not add any dependency.

Each --config is meant to be run in its OWN fresh process (see the launch
commands returned alongside this script) so CUDA allocator state and
compiled-kernel caches never leak between configs.

LS6 A100 results that motivated the configs below (all eval, full dims):
  dense_eager           OOM, peak allocated 37.481 GiB
  dense_compiled        passes cold + 2 warmed calls, peak allocated 6.850 GiB
  moe_eager             OOM, peak allocated 37.481 GiB
  moe_compiled          OOM, peak allocated 37.668 GiB
  moe_disable_boundary  OOM, peak allocated 37.481 GiB
All failures requested another 12.28 GiB at the softcap line
(`logits = 15 * logits * (logits.square() + 15**2).rsqrt()`), with only
~50 MiB reserved-but-unallocated at failure time -- i.e. not an allocator
fragmentation signature. dense_eager itself OOMs: whatever is closing the
~31 GiB gap between eager and compiled is not MoE-specific, it is something
`model.compile()` does for the WHOLE forward, dense or not.

Why moe_disable_boundary did not help (verified on CPU tracing against the
real GPT/Block/MoE classes, at toy dims, via TORCH_LOGS=graph_breaks -- see
the chat writeup for the exact log): wrapping MoE.forward in
torch._dynamo.disable() does stop Dynamo from tracing *into* the loop body's
problem op, but entering a disabled function is itself a graph-break event,
and that event still occurs inside GPT.forward's `for block in self.blocks:`
loop. Dynamo's "graph break in loop is not supported" rule then skips the
*containing frame*, which is GPT.forward itself -- not just the loop body.
Since GPT.forward's tail (norm2 -> proj -> float -> softcap -> cross_entropy)
lives in that same frame, it gets dragged into the eager fallback too, even
though the tail itself has no data-dependent shapes and nothing to do with
MoE. Confirmed directly in the log: "Developer debug context: frame skipped:
forward (...)" names GPT.forward, not MoE.forward.

New configs (dense_head_compiled / moe_head_compiled) test the fix this
implies: keep the block loop eager (so MoE's dispatch loop never has to
compile at all -- unchanged, unmodified), but pull the tail out into its own
top-level function and compile *that* in isolation with fullgraph=True, so
it is never in the same Dynamo frame as the blocks loop. model.compile() is
NOT called in these two configs -- only the standalone head_loss callable
is compiled, independently of the model object.

Configs:
  dense_eager             dense MLP, no torch.compile
  dense_compiled          dense MLP, model.compile(dynamic=False) (matches the trainer)
  moe_eager               existing MoE class, E=1,k=1, no compile
  moe_compiled            existing MoE class, E=1,k=1, model.compile(dynamic=False)
  moe_disable_boundary    existing MoE class, E=1,k=1, MoE.forward wrapped in
                          torch._dynamo.disable() (monkeypatched here, not in
                          the source file). Kept for the record: this does
                          NOT fix the OOM (see above) -- do not re-run
                          expecting a different result, it's evidence of the
                          mechanism, not a candidate fix.
  dense_head_compiled     dense MLP; embed/norm1/blocks run eager (plain
                          Python loop over model.blocks, unmodified
                          Block.forward); norm2/proj/float/softcap/
                          cross_entropy run inside a standalone function
                          compiled with torch.compile(fullgraph=True).
  moe_head_compiled       identical split, with the existing MoE(E=1,k=1)
                          class for the (still fully eager, untouched)
                          block loop.

GPU residency: the dense reference model is always constructed and
initialized on CPU. For a "dense_*" config it is moved to CUDA directly --
no MoE model is ever constructed. For a "moe_*" config, the MoE model is
built on CPU and has its non-router submodules copied from the CPU dense
model via load_state_dict (construction order differs between the two
models, which shifts the RNG stream, so matching seeds alone would not give
matching weights); the CPU dense reference is then deleted and
garbage-collected *before* the MoE model is moved to CUDA. At no point do
both models, or any leftover reference/state dict/temporary tensor, reside
on the GPU at once -- only the single model under test ever touches CUDA
memory.

A separate, CPU-only mode (--config head_loss_parity_check) validates the
isolated head/loss callable numerically: eager vs. torch.compile(fullgraph=
True) loss AND gradients (w.r.t. both the block-output activation and the
projection weight), using freshly randomized NONZERO projection weights.
The real training init zeros proj.weight, which would make an eager-vs-
compiled comparison numerically degenerate (all-zero logits) and is
explicitly not sufficient validation on its own. This mode needs no CUDA and
does not touch build_selected_model/run_call at all.

Interpretation note (deliberately not encoded as a conclusion anywhere in
this script): passing dense_head_compiled/moe_head_compiled establishes that
isolating the head/loss region from the (unmodified) block loop lets it
compile and recovers memory on that specific allocation -- a head/loss
memory workaround. It does not, by itself, show that a real E>1 multi-expert
model fits in memory; the block loop itself (where MoE dispatch lives)
remains uncompiled/eager in every config here, and its own memory behavior
at E>1 is untested by this script.
"""
import argparse
import gc
import traceback

import torch
import torch.nn.functional as F

from modded_nanogpt_moe import model as tgs

VOCAB_SIZE = 50304
NUM_LAYERS = 12
MODEL_DIM = 768
SEQ_LEN = 1024
MBS = 64  # matches the default trainer mbs -> 64*1024 = 65536 tokens/call

HEAD_COMPILED_CONFIGS = ("dense_head_compiled", "moe_head_compiled")
ALL_CONFIGS = ("dense_eager", "dense_compiled", "moe_eager", "moe_compiled",
               "moe_disable_boundary") + HEAD_COMPILED_CONFIGS


def init_dense_style(model):
    """Verbatim copy of the initialization loop in the active trainer."""
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")


def build_dense_cpu(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM):
    torch.manual_seed(0)
    model = tgs.GPT(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                     mlp_type="dense")
    init_dense_style(model)
    return model


def build_moe_matched_cpu(dense_model_cpu, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS,
                           model_dim=MODEL_DIM):
    """E=1,k=1 MoE model (on CPU) whose attention/norm/embed/proj/expert-0
    weights are copied verbatim from dense_model_cpu. Router weight is left
    at its own (valid) init: with num_experts=1 the softmax over a single
    logit is always exactly 1.0, so the router's value is provably
    irrelevant to the forward output -- only its presence costs a little
    extra compute/memory, which is exactly what this diagnostic wants to
    measure, not hide."""
    torch.manual_seed(0)
    moe_model = tgs.GPT(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                         mlp_type="moe", num_experts=1, top_k=1, normalize_topk=True)
    init_dense_style(moe_model)

    moe_model.embed.load_state_dict(dense_model_cpu.embed.state_dict())
    moe_model.proj.load_state_dict(dense_model_cpu.proj.state_dict())
    moe_model.norm1.load_state_dict(dense_model_cpu.norm1.state_dict())
    moe_model.norm2.load_state_dict(dense_model_cpu.norm2.state_dict())
    for db, mb in zip(dense_model_cpu.blocks, moe_model.blocks):
        mb.attn.load_state_dict(db.attn.state_dict())
        mb.norm1.load_state_dict(db.norm1.state_dict())
        mb.norm2.load_state_dict(db.norm2.state_dict())
        mb.mlp.experts[0].load_state_dict(db.mlp.state_dict())
    return moe_model


def build_selected_model(config):
    """Only ever moves ONE model to CUDA. The other reference is built (if
    needed at all), used for copying weights on CPU, then deleted and
    garbage-collected before anything touches the GPU."""
    dense_cpu = build_dense_cpu()

    if config.startswith("dense"):
        model_cpu = dense_cpu
    else:
        model_cpu = build_moe_matched_cpu(dense_cpu)
        del dense_cpu
        gc.collect()
        if config == "moe_disable_boundary":
            tgs.MoE.forward = torch._dynamo.disable(tgs.MoE.forward)

    model = model_cpu.cuda()
    del model_cpu
    gc.collect()

    if config in ("dense_compiled", "moe_compiled", "moe_disable_boundary"):
        model.compile(dynamic=False)
    # dense_head_compiled / moe_head_compiled: model.compile() is deliberately
    # NOT called here. Only the standalone head_loss callable (built in
    # main()) is compiled, independently of this model object -- see the
    # module docstring.
    return model


def eager_prefix(model, inputs):
    """Exactly GPT.forward's embed -> norm1 -> blocks section, unmodified
    and uncompiled. Block.forward (and MoE.forward, for moe_* configs) run
    exactly as they exist in the active model -- nothing about dispatch
    or routing is touched here."""
    x = model.norm1(model.embed(inputs))
    for block in model.blocks:
        x = block(x)
    return x


def make_head_loss(model):
    """Exactly GPT.forward's tail: norm2 -> proj -> float -> softcap ->
    cross_entropy(reduction='sum'), reusing model.norm2/model.proj directly
    (same nn.Parameter objects as the model under test -- no copies), so
    dtype behavior, parameter identity, and the loss reduction/scaling are
    byte-for-byte what GPT.forward does. No autocast context is introduced;
    the active trainer does not use one either."""
    def head_loss(x, targets):
        logits = model.proj(model.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
    return head_loss


def fixed_batch():
    torch.manual_seed(1234)
    inputs = torch.randint(0, VOCAB_SIZE, (MBS, SEQ_LEN), device="cuda", dtype=torch.int32)
    targets = torch.randint(0, VOCAB_SIZE, (MBS, SEQ_LEN), device="cuda", dtype=torch.int64)
    return inputs, targets


def mem_stats(label):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    ma = torch.cuda.max_memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    mr = torch.cuda.max_memory_reserved() / 2**30
    print(f"[{label}] allocated={a:.3f}GiB max_allocated={ma:.3f}GiB "
          f"reserved={r:.3f}GiB max_reserved={mr:.3f}GiB", flush=True)


def run_call(forward_fn, model, phase, label):
    # Pre-call snapshot (no reset) lets us see whether the baseline is
    # growing across calls, independent of the per-call peak below.
    mem_stats(label + " pre")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        loss = forward_fn()
        if phase == "backward":
            loss.backward()
        torch.cuda.synchronize()
        mem_stats(label + " OK")
    except Exception:
        print(f"[{label}] FAILED -- full traceback:", flush=True)
        traceback.print_exc()
        print(torch.cuda.memory_summary(), flush=True)
        mem_stats(label + " FAILED")
        raise
    finally:
        # Do not let the loss tensor (or, in the backward phase, the
        # autograd graph it roots) outlive this call.
        loss = None
        if phase == "backward":
            model.zero_grad(set_to_none=True)


def run_head_loss_parity_check():
    """CPU-only, no CUDA required. Numerical parity of the isolated
    head/loss callable: eager vs. torch.compile(fullgraph=True), checking
    BOTH the loss value and gradients w.r.t. the block-output activation and
    the projection weight, at small dimensions, with freshly randomized
    NONZERO projection weights (the real init zeros proj.weight, which would
    make this check numerically degenerate -- all-zero logits -- and is not
    sufficient validation by itself)."""
    small_vocab, small_layers, small_dim = 37, 2, 16
    tol = dict(atol=2e-2, rtol=2e-2)  # bf16-in/fp32-softcap tolerance

    torch.manual_seed(0)
    dense = build_dense_cpu(vocab_size=small_vocab, num_layers=small_layers, model_dim=small_dim)
    torch.manual_seed(1)
    dense.proj.weight.data.normal_(std=0.02)
    dense.proj.bias.data.normal_(std=0.02)
    moe = build_moe_matched_cpu(dense, vocab_size=small_vocab, num_layers=small_layers,
                                 model_dim=small_dim)

    all_ok = True
    for name, model in [("dense", dense), ("moe_E1k1", moe)]:
        torch.manual_seed(2)
        x_base = torch.randn(4, 6, small_dim, dtype=torch.bfloat16)
        targets = torch.randint(0, small_vocab, (4, 6))

        head_loss = make_head_loss(model)
        compiled_head_loss = torch.compile(head_loss, fullgraph=True, dynamic=False)

        for p in model.parameters():
            p.grad = None
        x_eager = x_base.clone().requires_grad_(True)
        loss_eager = head_loss(x_eager, targets)
        loss_eager.backward()
        grad_proj_eager = model.proj.weight.grad.clone()
        grad_x_eager = x_eager.grad.clone()

        for p in model.parameters():
            p.grad = None
        x_compiled = x_base.clone().requires_grad_(True)
        loss_compiled = compiled_head_loss(x_compiled, targets)
        loss_compiled.backward()
        grad_proj_compiled = model.proj.weight.grad.clone()
        grad_x_compiled = x_compiled.grad.clone()

        loss_ok = torch.allclose(loss_eager, loss_compiled, **tol)
        gproj_ok = torch.allclose(grad_proj_eager.float(), grad_proj_compiled.float(), **tol)
        gx_ok = torch.allclose(grad_x_eager.float(), grad_x_compiled.float(), **tol)
        all_ok = all_ok and loss_ok and gproj_ok and gx_ok

        print(f"[parity:{name}] loss eager={loss_eager.item():.6f} "
              f"compiled={loss_compiled.item():.6f} match={loss_ok}", flush=True)
        print(f"[parity:{name}] grad_proj max_abs_diff="
              f"{(grad_proj_eager.float()-grad_proj_compiled.float()).abs().max().item():.6f} "
              f"match={gproj_ok}", flush=True)
        print(f"[parity:{name}] grad_x    max_abs_diff="
              f"{(grad_x_eager.float()-grad_x_compiled.float()).abs().max().item():.6f} "
              f"match={gx_ok}", flush=True)
        assert not torch.allclose(grad_proj_eager, torch.zeros_like(grad_proj_eager)), (
            f"[parity:{name}] grad_proj is all zero -- this check is degenerate, "
            f"nonzero-weight override did not take effect")

    if not all_ok:
        print("PARITY CHECK FAILED", flush=True)
        raise SystemExit(1)
    print("PARITY CHECK PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                     choices=list(ALL_CONFIGS) + ["head_loss_parity_check"])
    ap.add_argument("--phase", choices=["eval", "backward"])
    ap.add_argument("--calls", type=int, default=3,
                     help="forward calls to make; call 0 = cold, rest = warmed")
    args = ap.parse_args()

    if args.config == "head_loss_parity_check":
        run_head_loss_parity_check()
        return

    if args.phase is None:
        raise SystemExit("--phase eval|backward is required for GPU configs")

    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"config={args.config} phase={args.phase} "
          f"vocab_size={VOCAB_SIZE} num_layers={NUM_LAYERS} model_dim={MODEL_DIM} "
          f"seq_len={SEQ_LEN} mbs={MBS} tokens_per_call={MBS * SEQ_LEN}", flush=True)

    model = build_selected_model(args.config)
    inputs, targets = fixed_batch()

    if args.config in HEAD_COMPILED_CONFIGS:
        head_loss = make_head_loss(model)
        compiled_head_loss = torch.compile(head_loss, fullgraph=True, dynamic=False)
        print(f"[independence check] model.compile() not called on the outer model; "
              f"only head_loss is compiled, as a standalone callable: "
              f"type(compiled_head_loss)={type(compiled_head_loss)}", flush=True)

        def forward_fn():
            x = eager_prefix(model, inputs)
            return compiled_head_loss(x, targets)
    else:
        def forward_fn():
            return model(inputs, targets)

    if args.phase == "eval":
        model.eval()
        grad_ctx = torch.no_grad()
    else:
        model.train()
        grad_ctx = torch.enable_grad()

    with grad_ctx:
        for i in range(args.calls):
            phase_word = "cold" if i == 0 else "warmed"
            label = f"{args.config}/{args.phase} call{i} {phase_word}"
            run_call(forward_fn, model, args.phase, label)


if __name__ == "__main__":
    main()
