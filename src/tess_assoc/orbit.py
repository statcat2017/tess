"""Orbital-alias inference, frozen formula layer (Phase 0).

Executable definition of Proposal §19: P_n = DeltaT / n for permitted
positive integers n, with the long-period lower boundary from protocol.

Defaults are intentionally bound at import time: protocol v1 is frozen,
so a v2 change means a new import. Do not "fix" these to None-sentinels.

Window-function filtering (Proposal §20) is a later phase.
"""

from __future__ import annotations

import math

from tess_assoc._validate import (
    require_finite,
    require_positive_finite,
    require_strict_int,
)
from tess_assoc.protocol import LONG_PERIOD_LOWER_BOUND_DAYS, MAX_ALIAS_N


def generate_aliases(
    delta_t_days: float,
    min_period_days: float = LONG_PERIOD_LOWER_BOUND_DAYS,
    max_n: int = MAX_ALIAS_N,
) -> list[float]:
    """Return [P_1, P_2, ...] with P_n = DeltaT/n and P_n >= min_period."""
    require_positive_finite("delta_t_days", delta_t_days)
    require_positive_finite("min_period_days", min_period_days)
    require_strict_int("max_n", max_n, minimum=1)
    n_max = min(max_n, math.floor(delta_t_days / min_period_days))
    if n_max < 1:
        raise ValueError("no aliases satisfy the period lower bound")
    return [delta_t_days / n for n in range(1, n_max + 1)]


def predict_epochs(t1: float, period_days: float, n_cycles: int) -> list[float]:
    """Predict t1 + k*P for k=0..n_cycles (increasing, preserves t1)."""
    require_finite("t1", t1)
    require_positive_finite("period_days", period_days)
    require_strict_int("n_cycles", n_cycles, minimum=0)
    return [t1 + k * period_days for k in range(n_cycles + 1)]
