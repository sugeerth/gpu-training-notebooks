"""Speculative decoding: what acceptance buys, and where it costs you.

Lifted verbatim from `Speculative_Decoding_Advanced_Serving.ipynb`.

The thing worth internalizing is that α enters as `α^(k+1)`, so a small change in acceptance is
a large change in how long a draft is worth running — which is why a schema-constrained tool
call, the most predictable text a model ever emits, gets a speedup prose cannot.
`kernels/18_spec_verify.cu` implements the verification and tabulates the regimes.
"""
from __future__ import annotations


def expected_tokens(alpha: float, k: int) -> float:
    """E[tokens emitted per verify round] at acceptance `alpha`, draft length `k`."""
    return (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1 else k + 1


def speedup(alpha: float, k: int, c: float = 0.05) -> float:
    """Wall-clock speedup including the draft's own cost.

    `c` is the draft pass cost as a fraction of a target pass. Below the break-even line this
    returns < 1: at low acceptance a longer draft makes you *slower*, because you pay for
    drafts that get rejected. That is the regime engines auto-tune away from, and the reason
    acceptance rate is the metric to watch rather than `k`.
    """
    return expected_tokens(alpha, k) / (k * c + 1)


def best_k(alpha: float, c: float = 0.05, kmax: int = 16) -> int:
    """The draft length that maximizes speedup at a given acceptance rate.

    Returns 0 when speculation does not pay at all, which happens more often than the
    literature's headline numbers suggest.
    """
    best, best_k_ = 1.0, 0
    for k in range(1, kmax + 1):
        s = speedup(alpha, k, c)
        if s > best:
            best, best_k_ = s, k
    return best_k_


__all__ = ["expected_tokens", "speedup", "best_k"]
