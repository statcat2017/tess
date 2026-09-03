"""Known-system replay through shared tracer stages (issue #3).

Builds a TracerManifest from real TESS-SPOC FFI extractions (predicted
transits of a known planet) and runs the same pairing, matching, alias,
and window-filter stages as the fixture path. Adds anchor bookkeeping
plus skipped-transit and provenance summaries on top of base results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import fields as _dc_fields
from typing import Any

from tess_assoc._validate import (
    require_finite,
    require_positive_finite,
    require_strict_int,
)
from tess_assoc.archive import ArchiveUnavailable, download_spoc_ffi
from tess_assoc.event import EventRecord
from tess_assoc.extract import (
    coverage_windows,
    extract_events,
    load_lightcurve,
    predicted_transits,
)
from tess_assoc.manifest import (
    ManifestEvent,
    ManifestSector,
    ReplayManifest,
    ReplaySystem,
    SYSTEM_REQUIRED_KEYS,
    TracerManifest,
)
from tess_assoc.pipeline import run_records
from tess_assoc.propose import (
    PROPOSER_SNR_THRESHOLD,
    Proposal,
    dip_snr_at,
    propose_with_detail,
    records_from_proposals,
)


RECALL_TOL_DAYS = 0.15

MISS_REASONS: tuple[str, ...] = (
    "proposed-unmeasurable",
    "no usable cadence",
    "fragmented by flagged cadences",
    "below-threshold",
)


@dataclass(frozen=True)
class MissedTransit:
    sector: int
    t0: float
    max_snr: float | None
    proposed: bool
    reason: str

    def __post_init__(self) -> None:
        require_strict_int("sector", self.sector, minimum=1)
        require_finite("t0", self.t0)
        if self.max_snr is not None:
            require_finite("max_snr", self.max_snr)
        if not isinstance(self.proposed, bool):
            raise ValueError("proposed must be a bool")
        if self.reason not in MISS_REASONS:
            raise ValueError(f"reason must be one of {list(MISS_REASONS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": int(self.sector),
            "t0": float(self.t0),
            "max_snr": None if self.max_snr is None else float(self.max_snr),
            "proposed": bool(self.proposed),
            "reason": self.reason,
        }


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


def _product_record(sector: int, product: ArchiveProduct) -> dict[str, Any]:
    return {
        "sector": sector,
        "data_uri": product.data_uri,
        "retrieved_utc": product.retrieved_utc,
        "cached": product.cached,
        "local_path": product.local_path,
    }


def _finish(
    *,
    name: str,
    tic_id: int,
    tol: float,
    thresholds: dict[str, float],
    manifest_sectors: list[ManifestSector],
    manifest_events: list[ManifestEvent],
    records: dict[str, EventRecord],
    skipped: list[dict[str, Any]],
    products: list[dict[str, Any]],
    anchors: list[str],
    extra: dict[str, Any] | None = None,
    records_runner=None,
) -> dict[str, Any]:
    """Shared tail: manifest build → stages → shaped results (one copy).

    records_runner defaults to the dev-gated run_records; the sealed
    holdout passes its freeze-gated runner instead. Same stages either way.
    """
    manifest = TracerManifest(
        name=name,
        tic_id=tic_id,
        epoch_match_tol_days=tol,
        matcher_thresholds=dict(thresholds),
        sectors=tuple(manifest_sectors),
        events=tuple(manifest_events),
    )
    results = (records_runner or run_records)(manifest, records)
    results["anchors"] = list(anchors)
    results["skipped"] = list(skipped)
    results["products"] = list(products)
    results["sectors"] = [
        {"sector": s.sector, "windows": [[a, b] for a, b in s.windows]}
        for s in manifest_sectors
    ]
    if extra:
        results.update(extra)
    return results


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
        products.append(_product_record(sector, product))
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

    by_sector: dict[int, list[str]] = {}
    for evid, rec in records.items():
        by_sector.setdefault(rec.sector, []).append(evid)
    return _finish(
        name=f"{replay.name}/{system.name}",
        tic_id=system.tic_id,
        tol=tol,
        thresholds=thresholds,
        manifest_sectors=manifest_sectors,
        manifest_events=manifest_events,
        records=records,
        skipped=skipped,
        products=products,
        anchors=[
            min(by_sector[s], key=lambda i: records[i].t0)
            for s in system.sectors
            if s in by_sector
        ],
        extra={"ephemeris_source": replay.ephemeris_source},
    )


def replay_all(
    path: str, cache_dir: str | None = None
) -> dict[str, dict[str, Any]]:
    replay = load_replay_manifest(path)
    return {
        system.name: replay_system(replay, system, cache_dir)
        for system in replay.systems
    }


def classify_pair(
    anchor_times: list[float],
    records: dict[str, EventRecord],
    pair_results: list[dict[str, Any]],
    tol_days: float = RECALL_TOL_DAYS,
) -> str:
    """Separate detection failure from association failure for a known pair."""
    require_positive_finite("tol_days", tol_days)
    near = [
        [rid for rid, rec in records.items() if abs(rec.t0 - t) <= tol_days]
        for t in anchor_times
    ]
    if not all(near):
        return "not-proposed"
    compatible = {
        (p["a"], p["b"]) for p in pair_results if p["compatible"]
    } | {(p["b"], p["a"]) for p in pair_results if p["compatible"]}
    if any((a, b) in compatible for a in near[0] for b in near[1]):
        return "associated"
    return "recalled-not-associated"


def replay_blind_system(
    replay: ReplayManifest, system: ReplaySystem, cache_dir: str | None = None,
    records_runner=None,
) -> dict[str, Any]:
    """Blind proposer path: no period, no ephemeris until recall scoring."""
    tol = replay.epoch_match_tol_days
    thresholds = dict(replay.matcher_thresholds)
    half_span = replay.window_half_span_days
    n_samples = replay.resample_samples

    records: dict[str, EventRecord] = {}
    manifest_sectors: list[ManifestSector] = []
    skipped: list[dict[str, Any]] = []
    n_proposals = 0
    products: list[dict[str, Any]] = []
    known: list[tuple[int, float]] = []
    anchor_times: list[float] = []
    sector_curves: dict[int, tuple[list[float], list[float], float]] = {}
    sector_proposals: dict[int, list[Proposal]] = {}
    for sector in system.sectors:
        product = download_spoc_ffi(system.tic_id, sector, cache_dir)
        products.append(_product_record(sector, product))
        time, flux = load_lightcurve(product)
        if not time:
            raise ArchiveUnavailable(f"no good cadences in {product.local_path}")
        sector_known = predicted_transits(
            system.t0_bjd_tdb, system.period_days, time[0], time[-1]
        )
        known.extend((sector, t) for t in sector_known)
        coverable = [
            t
            for t in sector_known
            if t - half_span >= time[0] and t + half_span <= time[-1]
        ]
        if coverable:
            anchor_times.append(coverable[0])
        proposals, detrended, sigma = propose_with_detail(time, flux)
        sector_curves[sector] = (time, detrended, sigma)
        sector_proposals[sector] = proposals
        n_proposals += len(proposals)
        recs, skipped_here = records_from_proposals(
            time,
            flux,
            proposals,
            tic_id=system.tic_id,
            sector=sector,
            half_span_days=half_span,
            resample_samples=n_samples,
            quality_base={"ephemeris_source": replay.ephemeris_source},
        )
        records.update(recs)
        skipped.extend(
            {"sector": sector, "t0": s.predicted_t0_btjd, "reason": s.reason}
            for s in skipped_here
        )
        manifest_sectors.append(
            ManifestSector(sector=sector, windows=tuple(coverage_windows(time)))
        )

    manifest_events = [
        ManifestEvent(
            id=rid,
            sector=rec.sector,
            t0=rec.t0,
            depth=rec.depth,
            duration_days=rec.duration_days,
            snr=rec.snr,
            shape="real",
            origin="proposal",
        )
        for rid, rec in records.items()
    ]
    recalled = [
        any(abs(rec.t0 - t) <= RECALL_TOL_DAYS for rec in records.values())
        for _, t in known
    ]
    coverable = [
        bool(
            t - half_span >= sector_curves[sec][0][0]
            and t + half_span <= sector_curves[sec][0][-1]
        )
        for sec, t in known
    ]
    recalled_coverable = sum(h and c for h, c in zip(recalled, coverable))
    missed: list[dict[str, Any]] = []
    search_half = system.duration_hours / 24.0
    for (sec, t), hit in zip(known, recalled):
        if hit:
            continue
        tcurve = sector_curves[sec]
        proposed = any(
            abs(p.t0_guess - t) <= RECALL_TOL_DAYS for p in sector_proposals[sec]
        )
        n_cad = sum(abs(x - t) <= search_half for x in tcurve[0])
        if n_cad == 0:
            missed.append(
                MissedTransit(
                    sector=sec, t0=t, max_snr=None,
                    proposed=proposed, reason="no usable cadence",
                ).to_dict()
            )
            continue
        max_snr = dip_snr_at(
            tcurve[0], tcurve[1], tcurve[2], t, search_half
        )
        if proposed:
            reason = "proposed-unmeasurable"
        elif max_snr >= PROPOSER_SNR_THRESHOLD:
            reason = "fragmented by flagged cadences"
        else:
            reason = "below-threshold"
        missed.append(
            MissedTransit(
                sector=sec, t0=t, max_snr=max_snr,
                proposed=proposed, reason=reason,
            ).to_dict()
        )
    if len(anchor_times) >= 2:
        anchor_ids = [anchor_times[0], anchor_times[-1]]
    else:
        anchor_ids = []
    results = _finish(
        name=f"{replay.name}/{system.name}/blind",
        tic_id=system.tic_id,
        tol=tol,
        thresholds=thresholds,
        manifest_sectors=manifest_sectors,
        manifest_events=manifest_events,
        records=records,
        skipped=skipped,
        products=products,
        anchors=[],
        records_runner=records_runner,
        extra={
            "n_proposals": n_proposals,
            "recall": {
                "known": len(known),
                "recalled": sum(recalled),
                "rate": (sum(recalled) / len(known)) if known else 0.0,
                "coverable": sum(coverable),
                "recalled_coverable": recalled_coverable,
                "rate_coverable": (recalled_coverable / sum(coverable))
                if sum(coverable)
                else 0.0,
            },
            "missed": missed,
            "ephemeris_source": replay.ephemeris_source,
        },
    )
    results["pair_outcome"] = (
        classify_pair(anchor_ids, records, results["pairs"])
        if len(anchor_ids) == 2
        else "not-proposed"
    )
    return results


__all__ = [
    "classify_pair",
    "load_replay_manifest",
    "replay_all",
    "replay_blind_system",
    "replay_system",
]
