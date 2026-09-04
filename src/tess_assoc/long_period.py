"""Period-free follow-up queue for isolated TESS transit-like events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tess_assoc._validate import require_finite, require_positive_finite
from tess_assoc.event import EventRecord


SINGLE_TRANSIT_SNR_THRESHOLD = 7.0
SINGLE_TRANSIT_MIN_BASELINE_DAYS = 180.0


def _value(event: EventRecord | Mapping[str, Any], key: str) -> Any:
    if isinstance(event, EventRecord):
        return getattr(event, key)
    try:
        return event[key]
    except KeyError as e:
        raise ValueError(f"single-transit event missing key: {key}") from e


def rank_single_transits(
    events: Iterable[EventRecord | Mapping[str, Any]],
    coverage_windows: Mapping[int, Sequence[tuple[float, float]]],
    *,
    min_snr: float = SINGLE_TRANSIT_SNR_THRESHOLD,
    min_baseline_days: float = SINGLE_TRANSIT_MIN_BASELINE_DAYS,
) -> list[dict[str, Any]]:
    """Rank stars with exactly one blind event over a long time baseline.

    A result has no period estimate: absence of a second event cannot prove a
    long period when the window function has gaps. The result is a follow-up
    queue, not an automatic planet promotion.
    """
    require_positive_finite("min_snr", min_snr)
    require_positive_finite("min_baseline_days", min_baseline_days)
    by_tic: dict[int, list[EventRecord | Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        tic_id = int(_value(event, "tic_id"))
        if tic_id < 1:
            raise ValueError("single-transit event tic_id must be positive")
        by_tic[tic_id].append(event)

    results = []
    for tic_id, tic_events in by_tic.items():
        if len(tic_events) != 1:
            continue
        event = tic_events[0]
        snr = float(_value(event, "snr"))
        require_finite("event snr", snr)
        if snr < min_snr:
            continue
        spans = [
            (float(start), float(end))
            for sector, sector_spans in coverage_windows.items()
            for start, end in sector_spans
            if end > start
        ]
        if not spans:
            continue
        baseline = max(end for _, end in spans) - min(start for start, _ in spans)
        if baseline < min_baseline_days:
            continue
        sector = int(_value(event, "sector"))
        if sector not in coverage_windows:
            continue
        results.append(
            {
                "tic_id": tic_id,
                "event": {
                    "sector": sector,
                    "t0": float(_value(event, "t0")),
                    "depth": float(_value(event, "depth")),
                    "duration_days": float(_value(event, "duration_days")),
                    "snr": snr,
                },
                "baseline_days": baseline,
                "covered_days": sum(end - start for start, end in spans),
                "period_status": "unconstrained",
                "score": snr,
            }
        )
    return sorted(results, key=lambda result: (-result["score"], result["tic_id"]))


__all__ = [
    "SINGLE_TRANSIT_MIN_BASELINE_DAYS",
    "SINGLE_TRANSIT_SNR_THRESHOLD",
    "rank_single_transits",
]
