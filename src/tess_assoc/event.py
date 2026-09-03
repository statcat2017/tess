"""Event record contract (Phase 0).

Enforces PRD User Stories 3-5: required fields, finite values,
time ordering, quality flags, serialization round trip.
Field names are owned by `tess_assoc.protocol.EVENT_REQUIRED_FIELDS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc._validate import (
    is_finite_number,
    is_strict_int,
    require_finite,
    require_positive_finite,
    require_strict_int,
)


@dataclass(frozen=True)
class EventRecord:
    tic_id: int
    sector: int
    t0: float  # event time, BTJD days
    local_time: tuple[float, ...] = ()  # days, strictly increasing
    local_flux: tuple[float, ...] = ()  # normalized, finite
    depth: float = 0.0  # fractional flux, > 0
    duration_days: float = 0.0  # days, > 0
    snr: float = 0.0  # > 0
    stellar_meta: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.local_time, (list, tuple)):
            raise ValueError("local_time must be a list/tuple")
        if not isinstance(self.local_flux, (list, tuple)):
            raise ValueError("local_flux must be a list/tuple")
        if not isinstance(self.stellar_meta, dict):
            raise ValueError("stellar_meta must be a dict")
        if not isinstance(self.quality, dict):
            raise ValueError("quality must be a dict")
        # Defensive copies so post-construction mutation is impossible.
        object.__setattr__(self, "local_time", tuple(self.local_time))
        object.__setattr__(self, "local_flux", tuple(self.local_flux))
        object.__setattr__(self, "stellar_meta", dict(self.stellar_meta))
        object.__setattr__(self, "quality", dict(self.quality))
        self.validate()

    def validate(self) -> None:
        require_strict_int("tic_id", self.tic_id, minimum=1)
        if not is_strict_int(self.sector) or self.sector not in _protocol.ALL_KNOWN_SECTORS:
            raise ValueError("sector must be a known TESS sector (1-106)")
        require_finite("t0", self.t0)
        require_positive_finite("depth", self.depth)
        require_positive_finite("duration_days", self.duration_days)
        require_positive_finite("snr", self.snr)
        if len(self.local_time) == 0 or len(self.local_flux) == 0:
            raise ValueError("local_time/local_flux must be non-empty")
        if len(self.local_time) != len(self.local_flux):
            raise ValueError("local_time and local_flux must have equal length")
        for v in (*self.local_time, *self.local_flux):
            if not is_finite_number(v):
                raise ValueError("local_time/local_flux must be finite")
        for a, b in zip(self.local_time, self.local_time[1:]):
            if not b > a:
                raise ValueError("local_time must be strictly increasing")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (list, tuple)):
                v = list(v)
            elif isinstance(v, dict):
                v = dict(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventRecord:
        if not isinstance(d, dict):
            raise ValueError("event payload must be a dict")
        missing = [k for k in _protocol.EVENT_REQUIRED_FIELDS if k not in d]
        if missing:
            raise ValueError(f"missing event fields: {missing}")
        extra = [k for k in d if k not in _protocol.EVENT_REQUIRED_FIELDS]
        if extra:
            raise ValueError(f"unknown event fields: {extra}")
        # Shape/value checks live in __post_init__/validate — just delegate.
        try:
            return cls(
                tic_id=d["tic_id"],
                sector=d["sector"],
                t0=d["t0"],
                local_time=d["local_time"],
                local_flux=d["local_flux"],
                depth=d["depth"],
                duration_days=d["duration_days"],
                snr=d["snr"],
                stellar_meta=d["stellar_meta"],
                quality=d["quality"],
            )
        except TypeError as e:
            raise ValueError(f"malformed event payload: {e}") from e
