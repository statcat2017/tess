"""Shared scalar validation (single dialect, no duplication).

Predicates for branch conditions, `require_*` helpers where a failed
check always raises — call sites use one line instead of an `if` pair.
"""

from __future__ import annotations

import math
from typing import Any


def is_strict_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def is_finite_number(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def require_finite(name: str, v: Any) -> None:
    if not is_finite_number(v):
        raise ValueError(f"{name} must be finite")


def require_positive_finite(name: str, v: Any) -> None:
    if not is_finite_number(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0")


def require_strict_int(name: str, v: Any, minimum: int | None = None) -> None:
    if not is_strict_int(v):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an int{suffix}")
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
