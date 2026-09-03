"""Known-system replay through shared tracer stages (issue #3).

Builds a TracerManifest from real TESS-SPOC FFI extractions (predicted
transits of a known planet) and runs the same pairing, matching, alias,
and window-filter stages as the fixture path. Adds anchor bookkeeping
plus skipped-transit and provenance summaries on top of base results.
"""

from __future__ import annotations

import json
from dataclasses import fields as _dc_fields
from typing import Any

from tess_assoc.archive import download_spoc_ffi
from tess_assoc.event import EventRecord
from tess_assoc.extract import extract_events
from tess_assoc.manifest import (
    ManifestEvent,
    ManifestSector,
    ReplayManifest,
    ReplaySystem,
    SYSTEM_REQUIRED_KEYS,
    TracerManifest,
)
from tess_assoc.pipeline import run_records


REPLAY_REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "product",
    "ephemeris_source",
    "epoch_match_tol_days",
    "window_half_span_days",
    "resample_samples",
    "matcher_thresholds",
    "systems",
)


def _load_system(d: dict[str, Any]) -> ReplaySystem:
    if not isinstance(d, dict):
        raise ValueError("each system must be a dict")
    for key in SYSTEM_REQUIRED_KEYS:
        if key not in d:
            raise ValueError(f"system missing key: {key}")
    allowed = {f.name for f in _dc_fields(ReplaySystem)}
    extra = [k for k in d if k not in allowed]
    if extra:
        raise ValueError(f"system unknown keys: {extra}")
    sectors = d["sectors"]
    if not isinstance(sectors, (list, tuple)):
        raise ValueError("system sectors must be a non-empty list")
    return ReplaySystem(
        name=d["name"],
        tic_id=d["tic_id"],
        period_days=d["period_days"],
        t0_bjd_tdb=d["t0_bjd_tdb"],
        duration_hours=d["duration_hours"],
        sectors=tuple(sectors),
        toi=d.get("toi", ""),
    )


def load_replay_manifest(path: str) -> ReplayManifest:
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("replay manifest must be a dict")
    for key in REPLAY_REQUIRED_KEYS:
        if key not in d:
            raise ValueError(f"replay manifest missing key: {key}")
    extra = [k for k in d if k not in REPLAY_REQUIRED_KEYS]
    if extra:
        raise ValueError(f"replay manifest unknown keys: {extra}")
    if not isinstance(d["matcher_thresholds"], dict):
        raise ValueError("matcher_thresholds must be a dict")
    if not isinstance(d["systems"], (list, tuple)):
        raise ValueError("systems must be a non-empty list")
    return ReplayManifest(
        name=d["name"],
        product=d["product"],
        ephemeris_source=d["ephemeris_source"],
        epoch_match_tol_days=d["epoch_match_tol_days"],
        window_half_span_days=d["window_half_span_days"],
        resample_samples=d["resample_samples"],
        matcher_thresholds=dict(d["matcher_thresholds"]),
        systems=tuple(_load_system(s) for s in d["systems"]),
    )


def replay_system(
    replay: ReplayManifest, system: ReplaySystem, cache_dir: str | None = None
) -> dict[str, Any]:
    """Replay one known system across its manifest sectors."""
    tol = replay.epoch_match_tol_days
    thresholds = dict(replay.matcher_thresholds)
    half_span = replay.window_half_span_days
    n_samples = replay.resample_samples

    records: dict[str, EventRecord] = {}
    manifest_events: list[ManifestEvent] = []
    manifest_sectors: list[ManifestSector] = []
    skipped: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    for sector in system.sectors:
        product = download_spoc_ffi(system.tic_id, sector, cache_dir)
        products.append(
            {
                "sector": sector,
                "data_uri": product.data_uri,
                "retrieved_utc": product.retrieved_utc,
                "cached": product.cached,
                "local_path": product.local_path,
            }
        )
        extracted, skipped_here, windows = extract_events(
            product, system, half_span, n_samples
        )
        manifest_sectors.append(
            ManifestSector(sector=sector, windows=tuple(windows))
        )
        for i, ext in enumerate(extracted):
            evid = f"S{sector}-{i:03d}"
            records[evid] = ext.record
            manifest_events.append(
                ManifestEvent(
                    id=evid,
                    sector=sector,
                    t0=ext.record.t0,
                    depth=ext.record.depth,
                    duration_days=ext.record.duration_days,
                    snr=ext.record.snr,
                    shape="real",
                    origin="ephemeris",
                )
            )
        skipped.extend(
            {"sector": sector, "t0": s.predicted_t0_btjd, "reason": s.reason}
            for s in skipped_here
        )

    manifest = TracerManifest(
        name=f"{replay.name}/{system.name}",
        tic_id=system.tic_id,
        epoch_match_tol_days=tol,
        matcher_thresholds=thresholds,
        sectors=tuple(manifest_sectors),
        events=tuple(manifest_events),
    )
    results = run_records(manifest, records)
    by_sector: dict[int, list[str]] = {}
    for evid, rec in records.items():
        by_sector.setdefault(rec.sector, []).append(evid)
    anchors = [
        min(by_sector[s], key=lambda i: records[i].t0)
        for s in system.sectors
        if s in by_sector
    ]
    results["anchors"] = anchors
    results["skipped"] = skipped
    results["products"] = products
    results["ephemeris_source"] = replay.ephemeris_source
    return results


def replay_all(
    path: str, cache_dir: str | None = None
) -> dict[str, dict[str, Any]]:
    replay = load_replay_manifest(path)
    return {
        system.name: replay_system(replay, system, cache_dir)
        for system in replay.systems
    }


__all__ = ["load_replay_manifest", "replay_system", "replay_all"]
