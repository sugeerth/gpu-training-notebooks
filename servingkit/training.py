"""Can you train it: memory per GPU, and what crosses the wire every step.

The four things that must live in memory at once — weights, gradients, optimizer states,
activations — plus the all-reduce that follows every step. ZeRO shards the first three;
FlashAttention deletes the quadratic term in the fourth; neither touches the last.

Lifted verbatim from `Training_Kernels_And_Memory.ipynb`, which is where the reasoning lives.
`demo/training-planner.html` reproduces it in JavaScript and `tools/verify_console.py` checks
the two agree over 2,592 configurations.
"""
from __future__ import annotations


def training_plan(params_b=7.0, recipe="bf16+master", gpus=8, vram_gb=80.0,
                  seq=4096, micro_batch=1, layers=32, hidden=4096, heads=32,
                  zero_stage=1, checkpointing="none", flash_attention=True,
                  link_gbps=450.0, link_lat_us=3.0):
    """Memory per GPU and communication per step for one training configuration.

    Returns a dict of GB figures plus the all-reduce cost. Everything is per-GPU except
    `grad_allreduce_gb`, which is the payload each GPU contributes.
    """
    P = params_b * 1e9

    # Bytes per parameter, by recipe. `w` folds together every copy of the weights that has
    # to exist at once — for mixed precision that is the bf16 copy the forward pass reads
    # plus the fp32 master the optimizer updates.
    RECIPES = {
        "fp32":         dict(w=4.0, g=4.0, m=4.0, v=4.0, trainable=1.0),
        "bf16+master":  dict(w=6.0, g=2.0, m=4.0, v=4.0, trainable=1.0),
        "bf16+8bit":    dict(w=6.0, g=2.0, m=1.0, v=1.0, trainable=1.0),
        "lora-r16":     dict(w=2.0, g=2.0, m=4.0, v=4.0, trainable=0.001),
    }
    r = RECIPES[recipe]
    t = r["trainable"]

    # ZeRO shards the *replicated* state across data-parallel ranks. Stage 1 the optimizer
    # states, stage 2 also the gradients, stage 3 also the parameters themselves.
    shard_opt = gpus if zero_stage >= 1 else 1
    shard_grad = gpus if zero_stage >= 2 else 1
    shard_param = gpus if zero_stage >= 3 else 1

    weights_gb = P * r["w"] / shard_param / 1e9
    grads_gb = P * t * r["g"] / shard_grad / 1e9
    optim_gb = P * t * (r["m"] + r["v"]) / shard_opt / 1e9

    # Activations, from Megatron-LM's accounting. Two terms per layer, at 16-bit:
    #
    #   s*b*h*34        the ordinary activations — linear in sequence length
    #   5*a*s^2*b       the attention score matrix and friends — QUADRATIC in s
    #
    # The second term is what FlashAttention deletes: it never materializes the score matrix,
    # so there is nothing to store. At s=4096, a=32 it is 2.7 GB per layer against 0.6 GB for
    # everything else, so whether it is present decides the entire activation budget. Nobody
    # trains long context without it, which is why it defaults to on here.
    base = seq * micro_batch * hidden * 34.0
    attn = 5.0 * heads * seq * seq * micro_batch
    per_layer = base + (0.0 if flash_attention else attn)
    if checkpointing == "full":
        # Store only each layer's input and recompute the rest: 2*s*b*h bytes per layer,
        # bought with roughly one extra forward pass (~33% more compute).
        per_layer = 2.0 * seq * micro_batch * hidden
    elif checkpointing == "selective":
        # Megatron's selective recompute: drop the quadratic term, keep the rest. With
        # FlashAttention this is already where you are, which is why the two coincide.
        per_layer = base
    act_gb = layers * per_layer / 1e9

    total_gb = weights_gb + grads_gb + optim_gb + act_gb
    fits = total_gb <= vram_gb

    # Gradient all-reduce. ZeRO-2 and above already reduce-scatter the gradients as part of
    # the sharding, so the payload is the same 2(R-1)/R factor either way.
    payload = P * t * r["g"]
    eff = 2.0 * (gpus - 1) / gpus if gpus > 1 else 0.0
    ring_steps = 2 * (gpus - 1) if gpus > 1 else 0
    comm_bytes = payload * eff
    ring_s = comm_bytes / (link_gbps * 1e9) + ring_steps * link_lat_us * 1e-6
    direct_s = comm_bytes / (link_gbps * 1e9) + 2 * link_lat_us * 1e-6

    # Compute, at a stated 40% model FLOPs utilization — the number a well-tuned run reaches
    # and the one worth planning against, rather than peak.
    tokens = seq * micro_batch * gpus
    recompute = 4.0 / 3.0 if checkpointing == "full" else 1.0
    flops = 6.0 * P * tokens * recompute

    return dict(
        weights_gb=weights_gb, grads_gb=grads_gb, optim_gb=optim_gb, act_gb=act_gb,
        total_gb=total_gb, vram_gb=vram_gb, fits=fits,
        headroom_gb=vram_gb - total_gb,
        grad_allreduce_gb=payload / 1e9, comm_gb=comm_bytes / 1e9,
        ring_ms=ring_s * 1e3, direct_ms=direct_s * 1e3, ring_steps=ring_steps,
        tokens_per_step=tokens, step_flops=flops,
        binding=("memory" if not fits else "compute"),
    )


__all__ = ["training_plan"]
