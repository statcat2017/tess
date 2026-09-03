"""Same-TIC candidate pairing (issue #2).

Unique unordered event pairs, no self-pairs, deterministic order.
"""

from __future__ import annotations

from dataclasses import dataclass

from tess_assoc.event import EventRecord


@dataclass(frozen=True)
class CandidatePair:
    a_id: str
    b_id: str
    tic_id: int


def build_pairs(events: dict[str, EventRecord]) -> list[CandidatePair]:
    if not events:
        return []
    tic_ids = {e.tic_id for e in events.values()}
    if len(tic_ids) != 1:
        raise ValueError("pairing requires a single TIC")
    tic_id = next(iter(tic_ids))
    ids = sorted(events, key=lambda i: (events[i].t0, i))
    pairs: list[CandidatePair] = []
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            pairs.append(CandidatePair(a_id=ids[x], b_id=ids[y], tic_id=tic_id))
    return pairs
