"""Compatibility entry point for the active dense/MoE trainer.

The implementation lives in the root-level ``modded_nanogpt_moe`` package.
This wrapper preserves the historical command and import surface used by the
track-3 tests and diagnostics.
"""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modded_nanogpt_moe.checkpoint import (  # noqa: E402,F401
    CHECKPOINT_FORMAT_VERSION,
    atomic_save_checkpoint,
    capture_rng_state,
    collect_environment_metadata,
    make_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    unwrap_model,
    validate_checkpoint_config,
)
from modded_nanogpt_moe.data import (  # noqa: E402,F401
    DistributedDataLoader,
    distributed_data_generator,
)
from modded_nanogpt_moe.model import (  # noqa: E402,F401
    Block,
    CausalSelfAttention,
    GPT,
    Linear,
    MLP,
    MoE,
    RMSNorm,
    Rotary,
    add_bias_by_expert_segments,
    eager_prefix,
    make_head_loss,
    resolve_mlp_hidden_dim,
)
from modded_nanogpt_moe.optim import (  # noqa: E402,F401
    Muon,
    build_optimizers,
    muon_update,
    zeropower_via_newtonschulz5,
)
from modded_nanogpt_moe.train import main, nsys_range  # noqa: E402,F401


if __name__ == "__main__":
    main()
