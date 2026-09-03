"""Phase 0 protocol freeze (v1).

Executable subset only — every value here is read by code.
Prose freeze (product, metrics, semantics, baselines) lives in
`protocol/FROZEN_v1.md`, not as string constants.

v1 values are immutable. Any change requires a new protocol version.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import Enum

from tess_assoc._validate import is_strict_int

PROTOCOL_VERSION = "v1"

# --- Sector boundaries (PRD + Proposal §21) ---
DEV_SECTORS: frozenset[int] = frozenset(range(1, 80))  # 1–79 inclusive
SEALED_SECTORS: frozenset[int] = frozenset(range(80, 106))  # 80–105 inclusive
DISCOVERY_SECTORS: frozenset[int] = frozenset({106})
ALL_KNOWN_SECTORS: frozenset[int] = DEV_SECTORS | SEALED_SECTORS | DISCOVERY_SECTORS

# --- Orbital inference (Proposal §19) ---
LONG_PERIOD_LOWER_BOUND_DAYS = 27.0
MAX_ALIAS_N = 10_000

# --- Event record contract (PRD User Stories 3-5) ---
# Enforced by tess_assoc.event.EventRecord.
EVENT_REQUIRED_FIELDS: tuple[str, ...] = (
    "tic_id",
    "sector",
    "t0",
    "local_time",
    "local_flux",
    "depth",
    "duration_days",
    "snr",
    "stellar_meta",
    "quality",
)


class SectorRole(Enum):
    DEV = "dev"
    SEALED = "sealed"
    DISCOVERY = "discovery"
    UNKNOWN = "unknown"


def sector_role(sector: int) -> SectorRole:
    """Single dispatch for sector membership (no branching at call sites)."""
    if not is_strict_int(sector):
        return SectorRole.UNKNOWN
    if sector in DEV_SECTORS:
        return SectorRole.DEV
    if sector in SEALED_SECTORS:
        return SectorRole.SEALED
    if sector in DISCOVERY_SECTORS:
        return SectorRole.DISCOVERY
    return SectorRole.UNKNOWN


_SEALED_OR_DISCOVERY = SEALED_SECTORS | DISCOVERY_SECTORS


def validate_no_temporal_leak(dev_sectors_used: Collection[int]) -> None:
    """Raise if any sealed/discovery sector leaks into development inputs."""
    leak = set(dev_sectors_used) & _SEALED_OR_DISCOVERY
    if leak:
        raise ValueError(f"temporal leak: sealed sectors in dev inputs: {sorted(leak)}")


def validate_tic_partition(*partitions: Collection[int]) -> None:
    """Raise if any TIC appears in more than one partition."""
    seen: set[int] = set()
    for part in partitions:
        overlap = seen & set(part)
        if overlap:
            raise ValueError(f"TIC partition leak: {sorted(overlap)[:5]}")
        seen |= set(part)
