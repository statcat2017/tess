"""Event record contract (Phase 0).

Implements PRD User Stories 3-5 + Testing Decisions:
required fields, units, finite values, time ordering,
cadence alignment (uniform sampling check is lenient in v1),
quality flags, serialization round trip.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EventRecord:
    tic_id: int
    sector: int
    t0: float  # event time, BTJD days
    local_time: list[float] = field(default_factory=list)  # days, strictly increasing
    local_flux: list[float] = field(default_factory=list)  # normalized, finite
    depth: float = 0.0  # fractional flux, > 0
    duration_days: float = 0.0  # days, > 0
    snr: float = 0.0  # > 0
    stellar_meta: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.tic_id, int) or self.tic_id <= 0:
            raise ValueError("tic_id must be a positive int")
        if not isinstance(self.sector, int) or self.sector < 1:
            raise ValueError("sector must be a positive int")
        for name in ("t0", "depth", "duration_days", "snr"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{name} must be finite")
        if not self.depth > 0:
            raise ValueError("depth must be > 0")
        if not self.duration_days > 0:
            raise ValueError("duration_days must be > 0")
        if not self.snr > 0:
            raise ValueError("snr must be > 0")
        if len(self.local_time) == 0 or len(self.local_flux) == 0:
            raise ValueError("local_time/local_flux must be non-empty")
        if len(self.local_time) != len(self.local_flux):
            raise ValueError("local_time and local_flux must have equal length")
        for v in list(self.local_time) + list(self.local_flux):
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError("local_time/local_flux must be finite")
        for a, b in zip(self.local_time, self.local_time[1:]):
            if not b > a:
                raise ValueError("local_time must be strictly increasing")
        if not isinstance(self.stellar_meta, dict):
            raise ValueError("stellar_meta must be a dict")
        if not isinstance(self.quality, dict):
            raise ValueError("quality must be a dict")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventRecord:
        rec = cls(
            tic_id=d["tic_id"],
            sector=d["sector"],
            t0=d["t0"],
            local_time=list(d["local_time"]),
            local_flux=list(d["local_flux"]),
            depth=d["depth"],
            duration_days=d["duration_days"],
            snr=d["snr"],
            stellar_meta=dict(d.get("stellar_meta", {})),
            quality=dict(d.get("quality", {})),
        )
        rec.validate()
        return rec
