"""Observing-window alias filter (issue #2, Proposal §20 tracer slice).

For each period alias of an associated pair, predict transit epochs across
the whole observed window span — before, between, and after the two events.
An alias predicting an epoch inside observed cadence with no event nearby
(within epoch_match_tol_days) is contradicted and rejected. The two
observed events themselves are never contradictions. Empty window coverage
cannot contradict anything, so every alias is retained.
"""

from __future__ import annotations

import bisect
import math
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


def _merge_windows(manifest: TracerManifest) -> tuple[list[float], list[float]]:
    spans = sorted((s, e) for sec in manifest.sectors for s, e in sec.windows)
    starts, ends = [], []
    for s, e in spans:
        if starts and s <= ends[-1]:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def filter_aliases(
    a: EventRecord,
    b: EventRecord,
    manifest: TracerManifest,
    events: list[EventRecord],
) -> list[AliasVerdict]:
    t1, t2 = sorted([a.t0, b.t0])
    delta_t = t2 - t1
    tol = manifest.epoch_match_tol_days
    starts, ends = _merge_windows(manifest)
    event_times = sorted(e.t0 for e in events)

    def observed(t: float) -> bool:
        i = bisect.bisect_right(starts, t) - 1
        return i >= 0 and t <= ends[i]

    def has_event(t: float) -> bool:
        i = bisect.bisect_left(event_times, t - tol)
        return i < len(event_times) and event_times[i] <= t + tol

    verdicts: list[AliasVerdict] = []
    for n, period in enumerate(generate_aliases(delta_t), start=1):
        contradiction: float | None = None
        if not starts:
            verdicts.append(
                AliasVerdict(n=n, period_days=period, retained=True)
            )
            continue
        k_min = math.ceil((starts[0] - t1) / period) - 1
        k_max = math.floor((ends[-1] - t1) / period) + 1
        for k in range(k_min, k_max + 1):
            epoch = t1 + k * period
            if abs(epoch - t1) <= tol or abs(epoch - t2) <= tol:
                continue  # observed anchors, never contradictions
            if observed(epoch) and not has_event(epoch):
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
