"""Orbital-alias inference, frozen formula layer (Phase 0).

Implements Proposal §19: P_n = DeltaT / n for permitted positive
integers n, with long-period lower boundary near 27 days.

Window-function filtering (Proposal §20) is a later phase;
this module covers only alias arithmetic + invariants so the
protocol freeze has an executable, tested definition.
"""

from __future__ import annotations

from tess_assoc.protocol import LONG_PERIOD_LOWER_BOUND_DAYS


def generate_aliases(
    delta_t_days: float,
    min_period_days: float = LONG_PERIOD_LOWER_BOUND_DAYS,
    max_n: int = 10_000,
) -> list[float]:
    """Return [P_1, P_2, ...] with P_n = DeltaT/n and P_n >= min_period."""
    if not delta_t_days > 0:
        raise ValueError("delta_t_days must be > 0")
    if not min_period_days > 0:
        raise ValueError("min_period_days must be > 0")
    aliases: list[float] = []
    n = 1
    while n <= max_n:
        p = delta_t_days / n
        if p < min_period_days:
            break
        aliases.append(p)
        n += 1
    if not aliases:
        raise ValueError("no aliases satisfy the period lower bound")
    return aliases


def predict_epochs(t1: float, period_days: float, n_cycles: int) -> list[float]:
    """Predict t1 + k*P for k=0..n_cycles (increasing, preserves t1)."""
    if not period_days > 0:
        raise ValueError("period_days must be > 0")
    if n_cycles < 0:
        raise ValueError("n_cycles must be >= 0")
    return [t1 + k * period_days for k in range(n_cycles + 1)]
