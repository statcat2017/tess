"""Fixture manifest loading and validation (issue #2).

A manifest is plain JSON describing one curated multiepoch system:
observing windows per sector plus ephemeris-anchored candidate events.
Every sector must be a development sector — sealed sectors are rejected
at load time so the tracer can never touch Sectors 80-105.

Payloads are strict: no type coercion, `ValueError` on any violation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc._validate import require_finite, require_positive_finite, require_strict_int
from tess_assoc.matcher import REQUIRED_THRESHOLDS


@dataclass(frozen=True)
class ManifestSector:
    sector: int
    windows: tuple[tuple[float, float], ...]  # (start, end) BTJD, sorted, disjoint

    def __post_init__(self) -> None:
        require_strict_int("sector", self.sector, minimum=1)
        if not isinstance(self.windows, (list, tuple)) or not self.windows:
            raise ValueError("sector windows must be a non-empty list")
        norm: list[tuple[float, float]] = []
        for w in self.windows:
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                raise ValueError("each window must be a [start, end] pair")
            require_finite("window start", w[0])
            require_finite("window end", w[1])
            if w[1] <= w[0]:
                raise ValueError("window end must be after start")
            norm.append((w[0], w[1]))
        if tuple(sorted(norm)) != tuple(norm):
            raise ValueError("sector windows must be sorted")
        for (s1, e1), (s2, e2) in zip(norm, norm[1:]):
            if s2 < e1:
                raise ValueError("sector windows must be disjoint")
        object.__setattr__(self, "windows", tuple(norm))


@dataclass(frozen=True)
class ManifestEvent:
    id: str
    sector: int
    t0: float
    depth: float
    duration_days: float
    snr: float
    shape: str  # "box" | "v" | "real"
    origin: str  # "ephemeris" | "distractor"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("event id must be a non-empty str")
        require_strict_int("event sector", self.sector, minimum=1)
        require_finite("event t0", self.t0)
        require_positive_finite("event depth", self.depth)
        require_positive_finite("event duration_days", self.duration_days)
        require_positive_finite("event snr", self.snr)
        if self.shape not in ("box", "v", "real"):
            raise ValueError("event shape must be 'box', 'v', or 'real'")
        if self.origin not in ("ephemeris", "distractor"):
            raise ValueError("event origin must be 'ephemeris' or 'distractor'")


@dataclass(frozen=True)
class TracerManifest:
    name: str
    tic_id: int
    epoch_match_tol_days: float
    matcher_thresholds: dict[str, float] = field(default_factory=dict)
    sectors: tuple[ManifestSector, ...] = ()
    events: tuple[ManifestEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("manifest name must be a non-empty str")
        require_strict_int("tic_id", self.tic_id, minimum=1)
        require_positive_finite("epoch_match_tol_days", self.epoch_match_tol_days)
        if not isinstance(self.matcher_thresholds, dict):
            raise ValueError("matcher_thresholds must be a dict")
        if not isinstance(self.sectors, (list, tuple)) or not all(
            isinstance(s, ManifestSector) for s in self.sectors
        ):
            raise ValueError("sectors must be ManifestSector records")
        if not isinstance(self.events, (list, tuple)) or not all(
            isinstance(e, ManifestEvent) for e in self.events
        ):
            raise ValueError("events must be ManifestEvent records")
        object.__setattr__(
            self, "matcher_thresholds", dict(self.matcher_thresholds)
        )
        object.__setattr__(self, "sectors", tuple(self.sectors))
        object.__setattr__(self, "events", tuple(self.events))
        if len({e.id for e in self.events}) != len(self.events):
            raise ValueError("event ids must be unique")


def _windows(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sector windows must be a non-empty list")
    out: list[tuple[float, float]] = []
    for w in value:
        if not isinstance(w, (list, tuple)) or len(w) != 2:
            raise ValueError("each window must be a [start, end] pair")
        require_finite("window start", w[0])
        require_finite("window end", w[1])
        if w[1] <= w[0]:
            raise ValueError("window end must be after start")
        out.append((w[0], w[1]))
    return tuple(sorted(out))


def load_manifest(d: dict[str, Any]) -> TracerManifest:
    if not isinstance(d, dict):
        raise ValueError("manifest must be a dict")
    for key in ("name", "tic_id", "sectors", "events", "matcher_thresholds"):
        if key not in d:
            raise ValueError(f"manifest missing key: {key}")
    if not isinstance(d["name"], str) or not d["name"]:
        raise ValueError("manifest name must be a non-empty str")
    require_strict_int("tic_id", d["tic_id"], minimum=1)
    tol = d.get("epoch_match_tol_days", 0.3)
    require_positive_finite("epoch_match_tol_days", tol)
    thresholds = d["matcher_thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("matcher_thresholds must be a dict")
    for key in REQUIRED_THRESHOLDS:
        if key not in thresholds:
            raise ValueError(f"matcher_thresholds missing key: {key}")
        require_finite(f"threshold {key}", thresholds[key])

    sectors: list[ManifestSector] = []
    for s in d["sectors"]:
        if not isinstance(s, dict) or "sector" not in s or "windows" not in s:
            raise ValueError("each sector needs 'sector' and 'windows'")
        sectors.append(
            ManifestSector(sector=s["sector"], windows=_windows(s["windows"]))
        )
    sector_ids = {s.sector for s in sectors}
    _protocol.validate_no_temporal_leak(sector_ids)
    by_sector = {s.sector: s for s in sectors}

    events = []
    for e in d["events"]:
        if not isinstance(e, dict):
            raise ValueError("each event must be a dict")
        for key in ("id", "sector", "t0", "depth", "duration_days", "snr"):
            if key not in e:
                raise ValueError(f"event missing key: {key}")
        require_strict_int("event sector", e["sector"], minimum=1)
        require_finite("event t0", e["t0"])
        if e["sector"] not in by_sector:
            raise ValueError(f"event sector {e['sector']} has no observing window")
        windows = by_sector[e["sector"]].windows
        if not any(s <= e["t0"] <= en for s, en in windows):
            raise ValueError(f"event {e.get('id')} t0 outside its sector windows")
        events.append(
            ManifestEvent(
                id=e["id"],
                sector=e["sector"],
                t0=e["t0"],
                depth=e["depth"],
                duration_days=e["duration_days"],
                snr=e["snr"],
                shape=e.get("shape", "box"),
                origin=e.get("origin", "ephemeris"),
            )
        )

    return TracerManifest(
        name=d["name"],
        tic_id=d["tic_id"],
        epoch_match_tol_days=tol,
        matcher_thresholds=dict(thresholds),
        sectors=tuple(sectors),
        events=tuple(events),
    )


def load_manifest_file(path: str) -> TracerManifest:
    with open(path) as f:
        return load_manifest(json.load(f))


SYSTEM_REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "tic_id",
    "period_days",
    "t0_bjd_tdb",
    "duration_hours",
    "sectors",
)


@dataclass(frozen=True)
class ReplaySystem:
    name: str
    tic_id: int
    period_days: float
    t0_bjd_tdb: float
    duration_hours: float
    sectors: tuple[int, ...] = ()
    toi: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("system name must be a non-empty str")
        require_strict_int("tic_id", self.tic_id, minimum=1)
        require_positive_finite("period_days", self.period_days)
        require_finite("t0_bjd_tdb", self.t0_bjd_tdb)
        require_positive_finite("duration_hours", self.duration_hours)
        if not isinstance(self.sectors, (list, tuple)) or not self.sectors:
            raise ValueError("system sectors must be a non-empty list")
        for sector in self.sectors:
            require_strict_int("sector", sector, minimum=1)
        if not isinstance(self.toi, str):
            raise ValueError("toi must be a str")
        object.__setattr__(self, "sectors", tuple(self.sectors))
        _protocol.validate_no_temporal_leak(set(self.sectors))


@dataclass(frozen=True)
class ReplayManifest:
    name: str
    product: str
    ephemeris_source: str
    epoch_match_tol_days: float
    window_half_span_days: float
    resample_samples: int
    matcher_thresholds: dict[str, float] = field(default_factory=dict)
    systems: tuple[ReplaySystem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("replay manifest name must be a non-empty str")
        if self.product != "TESS-SPOC FFI":
            raise ValueError("replay manifest must declare the TESS-SPOC FFI product")
        if not isinstance(self.ephemeris_source, str) or not self.ephemeris_source:
            raise ValueError("ephemeris_source must be a non-empty str")
        require_positive_finite("epoch_match_tol_days", self.epoch_match_tol_days)
        require_positive_finite("window_half_span_days", self.window_half_span_days)
        require_strict_int("resample_samples", self.resample_samples, minimum=3)
        if not isinstance(self.matcher_thresholds, dict):
            raise ValueError("matcher_thresholds must be a dict")
        for key in REQUIRED_THRESHOLDS:
            if key not in self.matcher_thresholds:
                raise ValueError(f"matcher_thresholds missing key: {key}")
            require_finite(f"threshold {key}", self.matcher_thresholds[key])
        if not isinstance(self.systems, (list, tuple)) or not self.systems:
            raise ValueError("systems must be a non-empty list")
        if not all(isinstance(s, ReplaySystem) for s in self.systems):
            raise ValueError("systems must be ReplaySystem records")
        object.__setattr__(
            self, "matcher_thresholds", dict(self.matcher_thresholds)
        )
        object.__setattr__(self, "systems", tuple(self.systems))
