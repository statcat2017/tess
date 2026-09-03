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


def _rel_diff(a: float, b: float) -> float:
    # Precondition: EventRecord guarantees depth/duration > 0, so mean > 0.
    return abs(a - b) / ((a + b) / 2.0)


def morphology_corr(a: EventRecord, b: EventRecord) -> float:
    """Pearson correlation of the local flux windows; 0.0 if undefined."""
    if len(a.local_flux) != len(b.local_flux) or not a.local_flux:
        return 0.0
    ma = sum(a.local_flux) / len(a.local_flux)
    mb = sum(b.local_flux) / len(b.local_flux)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a.local_flux, b.local_flux))
    va = sum((x - ma) ** 2 for x in a.local_flux)
    vb = sum((y - mb) ** 2 for y in b.local_flux)
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
    for key in ("max_rel_depth_diff", "max_rel_duration_diff", "min_morph_corr"):
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
