"""Deterministic association baseline (issue #2, PRD story 12-13).

Compares relative depth difference, relative duration difference,
normalized morphology correlation, and timing plausibility. Every
decision carries per-component values plus a human-readable explanation.
Thresholds come from the fixture manifest — never globals.
"""

from __future__ import annotations

from dataclasses import dataclass

from tess_assoc.event import EventRecord
from tess_assoc.orbit import generate_aliases


@dataclass(frozen=True)
class MatchDecision:
    compatible: bool
    rel_depth_diff: float
    rel_duration_diff: float
    morph_corr: float
    timing_plausible: bool
    explanation: str

    def __post_init__(self) -> None:
        # Cast numpy-adjacent scalars so decisions stay JSON-serializable.
        object.__setattr__(self, "compatible", bool(self.compatible))
        object.__setattr__(self, "rel_depth_diff", float(self.rel_depth_diff))
        object.__setattr__(self, "rel_duration_diff", float(self.rel_duration_diff))
        object.__setattr__(self, "morph_corr", float(self.morph_corr))
        object.__setattr__(self, "timing_plausible", bool(self.timing_plausible))
        object.__setattr__(self, "explanation", str(self.explanation))


REQUIRED_THRESHOLDS: tuple[str, ...] = (
    "max_rel_depth_diff",
    "max_rel_duration_diff",
    "min_morph_corr",
)


def _rel_diff(a: float, b: float) -> float:
    # Precondition: EventRecord guarantees depth/duration > 0, so mean > 0.
    return abs(a - b) / ((a + b) / 2.0)


def _detrend(flux: tuple[float, ...] | list[float]) -> list[float]:
    """Remove a least-squares line so sector-level slopes can't dominate shape."""
    n = len(flux)
    mean_x, mean_y = (n - 1) / 2.0, sum(flux) / n
    sxx = sum((x - mean_x) ** 2 for x in range(n))
    if sxx == 0:
        return [v - mean_y for v in flux]
    slope = sum((x - mean_x) * (v - mean_y) for x, v in enumerate(flux)) / sxx
    return [v - (mean_y + slope * (x - mean_x)) for x, v in enumerate(flux)]


def morphology_corr(a: EventRecord, b: EventRecord) -> float:
    """Pearson correlation of detrended local flux windows; 0.0 if undefined."""
    if len(a.local_flux) != len(b.local_flux) or not a.local_flux:
        return 0.0
    fa, fb = _detrend(a.local_flux), _detrend(b.local_flux)
    ma = sum(fa) / len(fa)
    mb = sum(fb) / len(fb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(fa, fb))
    va = sum((x - ma) ** 2 for x in fa)
    vb = sum((y - mb) ** 2 for y in fb)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va * vb) ** 0.5


def timing_plausible(a: EventRecord, b: EventRecord) -> bool:
    try:
        return bool(generate_aliases(abs(b.t0 - a.t0)))
    except ValueError:
        return False


def match(a: EventRecord, b: EventRecord, thresholds: dict[str, float]) -> MatchDecision:
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict")
    for key in REQUIRED_THRESHOLDS:
        if key not in thresholds:
            raise ValueError(f"thresholds missing key: {key}")
    depth_diff = _rel_diff(a.depth, b.depth)
    duration_diff = _rel_diff(a.duration_days, b.duration_days)
    corr = morphology_corr(a, b)
    timing = timing_plausible(a, b)
    depth_ok = depth_diff <= thresholds["max_rel_depth_diff"]
    duration_ok = duration_diff <= thresholds["max_rel_duration_diff"]
    morph_ok = corr >= thresholds["min_morph_corr"]
    compatible = depth_ok and duration_ok and morph_ok and timing
    parts = [
        f"depth diff {depth_diff:.3f} ({'ok' if depth_ok else 'FAIL'})",
        f"duration diff {duration_diff:.3f} ({'ok' if duration_ok else 'FAIL'})",
        f"morph corr {corr:.3f} ({'ok' if morph_ok else 'FAIL'})",
        f"timing ({'ok' if timing else 'FAIL'})",
    ]
    return MatchDecision(
        compatible=compatible,
        rel_depth_diff=depth_diff,
        rel_duration_diff=duration_diff,
        morph_corr=corr,
        timing_plausible=timing,
        explanation=("COMPATIBLE: " if compatible else "INCOMPATIBLE: ") + "; ".join(parts),
    )


def match_score(decision: MatchDecision) -> float:
    """Deterministic ranking score: timing gate, then shape minus mismatch.

    Timing-implausible pairs score -inf (never ranked above plausible ones).
    Compatible pairs (diffs <= ~0.25, corr >= ~0.9) always score >= ~0.65.
    """
    if not decision.timing_plausible:
        return float("-inf")
    return decision.morph_corr - max(
        decision.rel_depth_diff, decision.rel_duration_diff
    )
