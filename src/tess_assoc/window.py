"""Observing-window alias filter (issue #2, Proposal §20 tracer slice).

For each period alias of an associated pair, predict interior transit
epochs. An alias predicting an epoch inside observed cadence with no
event nearby (within tolerance) is contradicted and rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from tess_assoc.event import EventRecord
from tess_assoc.manifest import TracerManifest
from tess_assoc.orbit import generate_aliases


@dataclass(frozen=True)
class AliasVerdict:
    n: int
    period_days: float
    retained: bool
    contradicting_epoch: float | None = None


def _observed(t: float, manifest: TracerManifest) -> bool:
    return any(s <= t <= e for sec in manifest.sectors for s, e in sec.windows)


def _has_event(t: float, events: list[EventRecord], tol_days: float) -> bool:
    return any(abs(e.t0 - t) <= tol_days for e in events)


def filter_aliases(
    a: EventRecord,
    b: EventRecord,
    manifest: TracerManifest,
    events: list[EventRecord],
) -> list[AliasVerdict]:
    t1, t2 = sorted([a.t0, b.t0])
    delta_t = t2 - t1
    tol = manifest.epoch_match_tol_days
    verdicts: list[AliasVerdict] = []
    for n, period in enumerate(generate_aliases(delta_t), start=1):
        contradiction: float | None = None
        for k in range(1, n):
            epoch = t1 + k * period
            if _observed(epoch, manifest) and not _has_event(epoch, events, tol):
                contradiction = epoch
                break
        verdicts.append(
            AliasVerdict(
                n=n,
                period_days=period,
                retained=contradiction is None,
                contradicting_epoch=contradiction,
            )
        )
    return verdicts
