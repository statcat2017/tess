"""Fixture event-window provider (issue #2).

Builds deterministic local transit windows for manifest events — no blind
detection, no RNG. Identical ephemeris events share one analytic shape so
the matcher exercises morphology; the distractor uses a different shape,
depth, and duration. Provenance lands in `EventRecord.quality`.
"""

from __future__ import annotations

from tess_assoc.event import EventRecord
from tess_assoc.manifest import TracerManifest

N_SAMPLES = 61
HALF_SPAN_DAYS = 0.6


def _shape_flux(shape: str, phase: float, depth: float, duration_days: float) -> float:
    half = duration_days / 2.0
    if shape == "box":
        return 1.0 - depth if abs(phase) <= half else 1.0
    # "v": triangular dip over the same duration
    return 1.0 - depth * max(0.0, 1.0 - abs(phase) / half)


def provide_events(manifest: TracerManifest) -> dict[str, EventRecord]:
    step = (2.0 * HALF_SPAN_DAYS) / (N_SAMPLES - 1)
    out: dict[str, EventRecord] = {}
    for e in manifest.events:
        times = [e.t0 - HALF_SPAN_DAYS + i * step for i in range(N_SAMPLES)]
        fluxes = [
            _shape_flux(e.shape, t - e.t0, e.depth, e.duration_days) for t in times
        ]
        out[e.id] = EventRecord(
            tic_id=manifest.tic_id,
            sector=e.sector,
            t0=e.t0,
            local_time=times,
            local_flux=fluxes,
            depth=e.depth,
            duration_days=e.duration_days,
            snr=e.snr,
            stellar_meta={"r_star": 1.0},
            quality={
                "provider": "fixture",
                "fixture": manifest.name,
                "origin": e.origin,
                "shape": e.shape,
            },
        )
    return out
