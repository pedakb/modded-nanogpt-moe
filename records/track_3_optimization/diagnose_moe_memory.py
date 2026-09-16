"""
Standalone LS6 memory/compile diagnostic for the MoE(E=1,k=1) OOM.

Imports train_gpt_simple.py as a module WITHOUT editing it. Does not change
router/optimizer semantics, does not add an E=1 shortcut to the model code,
does not implement padded-bmm, does not add any dependency.

Each --config is meant to be run in its OWN fresh process (see the launch
commands returned alongside this script) so CUDA allocator state and
compiled-kernel caches never leak between configs.

Configs:
  dense_eager           dense MLP, no torch.compile
  dense_compiled         dense MLP, model.compile(dynamic=False)  (matches train_gpt_simple.py)
  moe_eager               existing MoE class, E=1,k=1, no compile
  moe_compiled            existing MoE class, E=1,k=1, model.compile(dynamic=False)
  moe_disable_boundary    existing MoE class, E=1,k=1, but MoE.forward is wrapped in
                          torch._dynamo.disable() (monkeypatched here, not in the
                          source file) so the surrounding model can still compile
                          instead of the whole frame being skipped. Expert execution
                          itself is byte-for-byte unchanged -- this only changes what
                          the compiler is allowed to see, not how experts run.

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

Interpretation note (deliberately not encoded as a conclusion anywhere in
this script): a memory/graph-break improvement from moe_disable_boundary
relative to moe_compiled would support a compilation-related memory
difference between the two. It would not, by itself, prove that allocator
fragmentation specifically is the mechanism, and a high peak-allocation
number alone does not identify which line of code (logits/softcap, MoE
dispatch, or something else) is responsible -- that requires reading the
actual traceback and torch.cuda.memory_summary() this script prints on
failure.
"""
import argparse
import gc
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt_simple as tgs  # noqa: E402

VOCAB_SIZE = 50304
NUM_LAYERS = 12
MODEL_DIM = 768
SEQ_LEN = 1024
MBS = 64  # matches train_gpt_simple.py's mbs -> 64*1024 = 65536 tokens/call


def init_dense_style(model):
    """Verbatim copy of the init loop in train_gpt_simple.py's __main__ block."""
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


def build_dense_cpu():
    torch.manual_seed(0)
    model = tgs.GPT(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
                     mlp_type="dense")
    init_dense_style(model)
    return model


def build_moe_matched_cpu(dense_model_cpu):
    """E=1,k=1 MoE model (on CPU) whose attention/norm/embed/proj/expert-0
    weights are copied verbatim from dense_model_cpu. Router weight is left
    at its own (valid) init: with num_experts=1 the softmax over a single
    logit is always exactly 1.0, so the router's value is provably
    irrelevant to the forward output -- only its presence costs a little
    extra compute/memory, which is exactly what this diagnostic wants to
    measure, not hide."""
    torch.manual_seed(0)
    moe_model = tgs.GPT(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
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
    return model


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


def run_call(model, inputs, targets, phase, label):
    # Pre-call snapshot (no reset) lets us see whether the baseline is
    # growing across calls, independent of the per-call peak below.
    mem_stats(label + " pre")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        loss = model(inputs, targets)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                     choices=["dense_eager", "dense_compiled", "moe_eager",
                              "moe_compiled", "moe_disable_boundary"])
    ap.add_argument("--phase", required=True, choices=["eval", "backward"])
    ap.add_argument("--calls", type=int, default=3,
                     help="forward calls to make; call 0 = cold, rest = warmed")
    args = ap.parse_args()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"config={args.config} phase={args.phase} "
          f"vocab_size={VOCAB_SIZE} num_layers={NUM_LAYERS} model_dim={MODEL_DIM} "
          f"seq_len={SEQ_LEN} mbs={MBS} tokens_per_call={MBS * SEQ_LEN}", flush=True)

    model = build_selected_model(args.config)
    inputs, targets = fixed_batch()

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
            run_call(model, inputs, targets, args.phase, label)


if __name__ == "__main__":
    main()
