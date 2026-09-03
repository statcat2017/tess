"""Injection-site geometry for the operating-regime study (issue #7).

Pure interval arithmetic: where a transit pair can be placed so both
event windows land on real cadences. No proposing, matching, or learning.
"""

from __future__ import annotations

import bisect

from tess_assoc._validate import require_positive_finite

DEPTHS: tuple[float, ...] = (0.003, 0.01, 0.025)
SHAPES: tuple[str, ...] = ("box", "v")
SAME_EPOCH_DT_DAYS = 12.0
INJECTION_DURATION_DAYS = 0.12


def feasible_interval(
    span_a: tuple[float, float],
    span_b: tuple[float, float],
    delta_t_days: float,
    half_span_days: float,
) -> tuple[float, float] | None:
    """Feasible t1 range with full windows for t1 in A and t1+delta in B."""
    require_positive_finite("delta_t_days", delta_t_days)
    lo = max(span_a[0] + half_span_days, span_b[0] + half_span_days - delta_t_days)
    hi = min(span_a[1] - half_span_days, span_b[1] - half_span_days - delta_t_days)
    if hi < lo:
        return None
    return (lo, hi)


def feasible_pair(
    span_a: tuple[float, float],
    span_b: tuple[float, float],
    delta_t_days: float,
    half_span_days: float,
) -> float | None:
    """Midpoint t1 with full windows for t1 in A and t1+delta in B, else None."""
    interval = feasible_interval(span_a, span_b, delta_t_days, half_span_days)
    if interval is None:
        return None
    return (interval[0] + interval[1]) / 2.0


def _count_in_window(sorted_time: list[float], center: float, half_span: float) -> int:
    lo = bisect.bisect_left(sorted_time, center - half_span)
    hi = bisect.bisect_right(sorted_time, center + half_span)
    return hi - lo


def supported_pair(
    time_a: list[float],
    time_b: list[float],
    delta_t_days: float,
    half_span_days: float,
    min_points: int = 20,
) -> float | None:
    """Feasible t1 whose windows hold enough real cadences (never gaps).

    Scans 33 evenly spaced candidates and keeps the best-supported one
    (most limiting-side cadences, then most total, then nearest midpoint).
    Counting is O(log N) per candidate via bisection on sorted cadences.
    """
    if not time_a or not time_b:
        return None
    interval = feasible_interval(
        (time_a[0], time_a[-1]), (time_b[0], time_b[-1]),
        delta_t_days, half_span_days,
    )
    if interval is None:
        return None
    sorted_a = sorted(time_a)
    sorted_b = sorted(time_b)
    lo, hi = interval
    midpoint = (lo + hi) / 2.0
    best: tuple[int, int, float, float] | None = None
    for k in range(33):
        t1 = lo + (hi - lo) * k / 32.0
        t2 = t1 + delta_t_days
        n_a = _count_in_window(sorted_a, t1, half_span_days)
        n_b = _count_in_window(sorted_b, t2, half_span_days)
        if n_a >= min_points and n_b >= min_points:
            score = (min(n_a, n_b), n_a + n_b, -abs(t1 - midpoint), t1)
            if best is None or score > best:
                best = score
    return best[3] if best is not None else None


__all__ = [
    "DEPTHS",
    "SHAPES",
    "SAME_EPOCH_DT_DAYS",
    "INJECTION_DURATION_DAYS",
    "feasible_interval",
    "feasible_pair",
    "supported_pair",
]
