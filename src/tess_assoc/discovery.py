"""Sector 106 discovery cohort (issue #9).

Applies the frozen pipeline to new data: blind proposal in early and
Sector 106 light curves, cross-epoch (early x 106) pair ranking by the
frozen deterministic score, alias filtering, automated vetting, and a
promotion rule that admits only clean pairs. Anything without Sector 106
runs as a labeled rehearsal — full machinery, zero promotions.

Sealed sectors (80-105) are rejected outright: the discovery cohort is
early sectors plus Sector 106 only. Alternate reductions and external
photometry stay manual, candidate-gated steps, recorded — never fetched.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any

from tess_assoc import freeze as _freeze
from tess_assoc import protocol as _protocol
from tess_assoc._validate import (
    require_finite,
    require_positive_finite,
    require_strict_int,
)
from tess_assoc.archive import ArchiveUnavailable
from tess_assoc.event import EventRecord
from tess_assoc.extract import load_lightcurve, predicted_transits
from tess_assoc.matcher import REQUIRED_THRESHOLDS, match, match_score
from tess_assoc.propose import propose_with_detail
from tess_assoc.pipeline import run_frozen_records
from tess_assoc.replay import RECALL_TOL_DAYS, replay_blind_system
from tess_assoc.vetting import (
    check_companion_radius,
    check_contamination,
    check_variables,
    combine_secondary_searches,
    cross_match_toi,
    promote_candidate,
    secondary_search,
    stellar_radius,
)
from tess_assoc.window import filter_aliases

DISCOVERY_SECTOR = 106


def _allowed_sectors() -> frozenset[int]:
    return _protocol.DEV_SECTORS | {DISCOVERY_SECTOR}


@dataclass(frozen=True)
class DiscoverySystem:
    """Cohort target: early sectors plus Sector 106 (never sealed)."""

    name: str
    tic_id: int
    period_days: float | None = None
    t0_bjd_tdb: float | None = None
    duration_hours: float | None = None
    sectors: tuple[int, ...] = ()
    toi: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("system name must be a non-empty str")
        require_strict_int("tic_id", self.tic_id, minimum=1)
        if self.period_days is not None:
            require_positive_finite("period_days", self.period_days)
        if self.t0_bjd_tdb is not None:
            require_finite("t0_bjd_tdb", self.t0_bjd_tdb)
        if self.duration_hours is not None:
            require_positive_finite("duration_hours", self.duration_hours)
        if not isinstance(self.sectors, (list, tuple)) or not self.sectors:
            raise ValueError("system sectors must be a non-empty list")
        for sector in self.sectors:
            require_strict_int("sector", sector, minimum=1)
            if sector not in _allowed_sectors():
                raise ValueError(
                    f"sector {sector} not allowed in discovery cohort "
                    "(early sectors + 106 only; sealed sectors excluded)"
                )
        if not isinstance(self.toi, str):
            raise ValueError("toi must be a str")
        object.__setattr__(self, "sectors", tuple(self.sectors))


@dataclass(frozen=True)
class DiscoveryManifest:
    """Discovery cohort manifest (duck-types ReplayManifest for blind replay).

    purpose selects the honesty regime: "rehearsal" (stand-in data, never
    promoted), "mining" (archive multi-epoch data, promotions allowed and
    labeled), "discovery" (requires Sector 106).
    """

    name: str
    product: str
    ephemeris_source: str
    epoch_match_tol_days: float
    window_half_span_days: float
    resample_samples: int
    matcher_thresholds: dict[str, float] = field(default_factory=dict)
    systems: tuple[DiscoverySystem, ...] = ()
    purpose: str = "rehearsal"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("discovery manifest name must be a non-empty str")
        if self.product != "TESS-SPOC FFI":
            raise ValueError("discovery manifest must declare the TESS-SPOC FFI product")
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
        if not all(isinstance(s, DiscoverySystem) for s in self.systems):
            raise ValueError("systems must be DiscoverySystem records")
        if self.purpose not in ("rehearsal", "mining", "discovery"):
            raise ValueError("purpose must be rehearsal, mining, or discovery")
        has_new = any(DISCOVERY_SECTOR in s.sectors for s in self.systems)
        if self.purpose == "discovery" and not has_new:
            raise ValueError("discovery purpose requires Sector 106 in cohort")
        if self.purpose == "mining" and has_new:
            raise ValueError("mining purpose excludes Sector 106 (that is discovery)")
        if len({s.name for s in self.systems}) != len(self.systems):
            raise ValueError("system names must be unique (results key on name)")
        object.__setattr__(self, "matcher_thresholds", dict(self.matcher_thresholds))
        object.__setattr__(self, "systems", tuple(self.systems))


def _parse_discovery_manifest(d: dict[str, Any]) -> DiscoveryManifest:
    for key in (
        "name", "product", "ephemeris_source", "epoch_match_tol_days",
        "window_half_span_days", "resample_samples", "matcher_thresholds",
        "systems",
    ):
        if key not in d:
            raise ValueError(f"discovery manifest missing key: {key}")
    systems = [
        DiscoverySystem(
            name=s["name"],
            tic_id=s["tic_id"],
            period_days=s.get("period_days"),
            t0_bjd_tdb=s.get("t0_bjd_tdb"),
            duration_hours=s.get("duration_hours"),
            sectors=tuple(s["sectors"]),
            toi=s.get("toi", ""),
        )
        for s in d["systems"]
    ]
    return DiscoveryManifest(
        name=d["name"],
        product=d["product"],
        ephemeris_source=d["ephemeris_source"],
        epoch_match_tol_days=d["epoch_match_tol_days"],
        window_half_span_days=d["window_half_span_days"],
        resample_samples=d["resample_samples"],
        matcher_thresholds=dict(d["matcher_thresholds"]),
        systems=tuple(systems),
        purpose=d.get("purpose", "rehearsal"),
    )


def load_discovery_manifest(
    path: str, freeze_record, config
) -> DiscoveryManifest:
    """Discovery gate: verified freeze required; sealed sectors never load."""
    record = _freeze.verify_freeze(freeze_record, config)
    _freeze.check_manifest_bytes(path, record, "discovery")
    with open(path) as f:
        manifest = _parse_discovery_manifest(json.load(f))
    if dict(manifest.matcher_thresholds) != record.thresholds:
        raise ValueError("discovery thresholds differ from frozen thresholds")
    return manifest


def cone_tics(
    ra_deg: float, dec_deg: float, radius_deg: float, mag_limit: float
) -> list[dict[str, Any]]:
    """Bright TICs in a sky cone (cohort-selection primitive, metadata only)."""
    from astroquery.mast import Catalogs

    try:
        tab = Catalogs.query_region(
            f"{ra_deg} {dec_deg}",
            radius=f"{radius_deg} deg",
            catalog="Tic",
        )
    except Exception as e:
        raise ArchiveUnavailable(f"TIC cone query failed: {e}") from e
    out = []
    for r in tab:
        try:
            tmag = float(r["Tmag"])
        except (TypeError, ValueError):
            continue
        if tmag <= mag_limit:
            out.append({"tic_id": int(r["ID"]), "tmag": tmag})
    return sorted(out, key=lambda d: d["tmag"])


def sectors_for_tic(tic_id: int) -> list[int]:
    """Archived TESS-SPOC FFI sectors for one TIC (metadata only)."""
    from astroquery.mast import Observations

    try:
        rows = Observations.query_criteria(
            target_name=str(tic_id),
            obs_collection="HLSP",
            dataproduct_type="timeseries",
        )
    except Exception as e:
        raise ArchiveUnavailable(f"MAST coverage query failed for TIC {tic_id}: {e}") from e
    return sorted(
        {
            int(r["sequence_number"])
            for r in rows
            if str(r["provenance_name"]) == "TESS-SPOC"
        }
    )


def select_cohort(
    ra_deg: float,
    dec_deg: float,
    target_sector: int,
    *,
    radius_deg: float = 0.5,
    mag_limit: float = 12.0,
    max_targets: int = 10,
) -> list[dict[str, Any]]:
    """TICs in a footprint cone holding target + early-sector SPOC data."""
    require_strict_int("target_sector", target_sector, minimum=1)
    cohort = []
    for entry in cone_tics(ra_deg, dec_deg, radius_deg, mag_limit):
        sectors = sectors_for_tic(entry["tic_id"])
        early = [s for s in sectors if s in _protocol.DEV_SECTORS]
        if target_sector in sectors and early:
            cohort.append({**entry, "sectors": sectors, "early_sectors": early})
        if len(cohort) >= max_targets:
            break
    return cohort


def _vetting_inputs(
    products: list[dict[str, Any]], tic_id: int, sector: int
) -> tuple[list[float], list[float], list[float], float | None]:
    """Reload cached sector curve + detrending for secondary search."""
    from tess_assoc.archive import ArchiveProduct

    matches = [p for p in products if p["sector"] == sector]
    if not matches:
        return [], [], [], None
    product = ArchiveProduct(
        tic_id=tic_id,
        sector=sector,
        local_path=matches[0]["local_path"],
        data_uri=matches[0].get("data_uri", ""),
        retrieved_utc=matches[0].get("retrieved_utc", ""),
        cached=True,
    )
    time, flux = load_lightcurve(product)
    _, detrended, sigma = propose_with_detail(time, flux)
    return time, flux, detrended, sigma


def _vet_pair(
    tic_id: int,
    event_a: dict[str, Any],
    event_b: dict[str, Any],
    retained_periods: list[float],
    products: list[dict[str, Any]],
    half_span_days: float,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Automated vetting for one ranked pair (catalog queries are live).

    Secondary search spans BOTH event sectors: an EB secondary landing in
    either window must flag. Per-TIC catalog lookups come from the shared
    cache (one MAST/TAP query per star, not per pair). Events are plain
    summaries so harvested pairs stay JSON-serializable (survey resume).
    """
    if catalog is None:
        catalog = {
            "contamination": check_contamination(tic_id),
            "cross_match": cross_match_toi(tic_id),
            "stellar_rad": stellar_radius(tic_id),
            "variables": check_variables(tic_id),
        }
    if "stellar_rad" not in catalog:
        catalog["stellar_rad"] = stellar_radius(tic_id)
    if "variables" not in catalog:
        catalog["variables"] = check_variables(tic_id)
    results: list[dict[str, Any]] = []
    for event in (event_a, event_b):
        time_r, _, detrended_r, sigma_r = _vetting_inputs(
            products, tic_id, event["sector"]
        )
        if not time_r or not sigma_r:
            continue
        results.append(
            secondary_search(
                time_r, detrended_r, sigma_r, event["t0"],
                retained_periods, half_span_days,
            )
        )
    secondary = combine_secondary_searches(retained_periods, results)
    companion = check_companion_radius(
        tic_id,
        max(event_a["depth"], event_b["depth"]),
        rad=catalog.get("stellar_rad"),
    )
    return {
        "secondary": secondary,
        "contamination": catalog["contamination"],
        "cross_match": catalog["cross_match"],
        "companion": companion,
        "variables": catalog["variables"],
    }


def harvest_system(
    manifest: DiscoveryManifest,
    system: DiscoverySystem,
    *,
    record,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Blind replay + cross-epoch pairs for one cohort system.

    Fault-isolated unit: ArchiveUnavailable becomes a blocked payload,
    never an exception. Shared by run_discovery and the survey runner.
    """
    runner = functools.partial(run_frozen_records, freeze_record=record)
    try:
        res = replay_blind_system(manifest, system, cache_dir, records_runner=runner)
    except ArchiveUnavailable as e:
        return {
            "status": "blocked-on-archive",
            "systems_out": {
                "tic_id": system.tic_id,
                "sectors": list(system.sectors),
                "status": "blocked-on-archive",
                "reason": str(e),
            },
            "blind_result": None,
            "pairs": [],
        }
    records = _records_of(res)
    windows = {
        entry["sector"]: [tuple(w) for w in entry["windows"]]
        for entry in res["sectors"]
    }
    pairs = _cross_epoch_pairs(records, windows, dict(manifest.matcher_thresholds))
    return {
        "status": "complete",
        "systems_out": {
            "tic_id": system.tic_id,
            "sectors": [s["sector"] for s in res["sectors"]],
            "status": "complete",
            "n_proposals": res["n_proposals"],
            "recall": res["recall"],
            "pair_outcome": res["pair_outcome"],
            "n_cross_pairs": len(pairs),
        },
        "blind_result": res,
        "products": res["products"],
        "pairs": pairs,
    }


def triage_ranked_pairs(
    manifest: DiscoveryManifest,
    harvests: dict[str, dict[str, Any]],
    *,
    shortlist_k: int = 10,
    per_system_cap: int = 5,
    catalog_prefetch: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Global rank + vetting + promotion over harvested pairs.

    catalog_prefetch maps tic_id -> {"contamination": ..., "cross_match": ...}
    (batched survey lookups); missing TICs fall back to live queries.
    Per-system cap keeps one busy star from flooding the review budget.
    Returns (candidates, reviewed).
    """
    half_span = manifest.window_half_span_days
    allow_promotion = manifest.purpose != "rehearsal"
    by_system: dict[str, list[dict[str, Any]]] = {}
    for name, harvest in harvests.items():
        by_system[name] = sorted(
            harvest["pairs"], key=lambda pair: pair["score"], reverse=True
        )[:per_system_cap]
    ranked = sorted(
        (
            (name, pair)
            for name, pairs in by_system.items()
            for pair in pairs
        ),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    candidates: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    catalog_cache: dict[int, dict[str, Any]] = dict(catalog_prefetch or {})
    by_name = {s.name: s for s in manifest.systems}
    for name, pair in ranked[:shortlist_k]:
        system = by_name[name]
        blind_products = harvests[name]["products"]
        if system.tic_id not in catalog_cache:
            catalog_cache[system.tic_id] = {
                "contamination": check_contamination(system.tic_id),
                "cross_match": cross_match_toi(system.tic_id),
                "stellar_rad": stellar_radius(system.tic_id),
                "variables": check_variables(system.tic_id),
            }
        vetting = _vet_pair(
            system.tic_id, pair["event_a"], pair["event_b"],
            pair["retained_periods"], blind_products, half_span,
            catalog=catalog_cache[system.tic_id],
        )
        promotion = promote_candidate(
            compatible=pair["compatible"],
            aliases_retained=len(pair["retained_periods"]),
            cross_match=vetting["cross_match"],
            contamination=vetting["contamination"],
            secondary=vetting["secondary"],
            companion=vetting["companion"],
            variables=vetting["variables"],
        )
        entry = {
            "system": name,
            "tic_id": system.tic_id,
            "record_a": pair["a"],
            "record_b": pair["b"],
            "event_a": pair["event_a"],
            "event_b": pair["event_b"],
            "score": pair["score"],
            "morph_corr": pair["morph_corr"],
            "retained_periods": pair["retained_periods"],
            "vetting": vetting,
            "promotion": promotion,
            "alternate_reduction_check": "pending-manual",
            "external_photometry": "pending-manual",
            "products": [
                p for p in harvests[name]["products"]
                if p["sector"] in (pair["event_a"]["sector"], pair["event_b"]["sector"])
            ],
        }
        if not allow_promotion:
            entry["promotion"] = {
                "candidate": False,
                "reasons": ["rehearsal cohort (promotion disabled)"],
                "manual_checklist": promotion["manual_checklist"],
            }
        (candidates if entry["promotion"]["candidate"] else reviewed).append(entry)
    return candidates, reviewed


def run_discovery(
    manifest: DiscoveryManifest,
    *,
    freeze_path: str,
    config,
    cache_dir: str | None = None,
    log_path: str | None = None,
    shortlist_k: int = 10,
) -> dict[str, Any]:
    """Frozen discovery run over the cohort (rehearsal if no Sector 106)."""
    record = _freeze.verify_freeze(freeze_path, config)
    if dict(manifest.matcher_thresholds) != record.thresholds:
        raise ValueError("discovery thresholds differ from frozen thresholds")
    record = _freeze.mark_unblinded(freeze_path)
    is_discovery = any(DISCOVERY_SECTOR in s.sectors for s in manifest.systems)

    systems_out: dict[str, Any] = {}
    blocked: list[str] = []
    harvests: dict[str, dict[str, Any]] = {}
    for system in manifest.systems:
        harvest = harvest_system(manifest, system, record=record, cache_dir=cache_dir)
        systems_out[system.name] = harvest["systems_out"]
        if harvest["status"] == "blocked-on-archive":
            blocked.append(system.name)
        else:
            harvests[system.name] = harvest

    candidates, reviewed = triage_ranked_pairs(
        manifest, harvests, shortlist_k=shortlist_k
    )
    blind_results = {name: h["blind_result"] for name, h in harvests.items()}
    n_pairs_ranked = sum(len(h["pairs"]) for h in harvests.values())

    status = "complete"
    if blocked and not blind_results:
        status = "blocked-on-archive"
    elif blocked:
        status = "partial"
    sealed = sorted(
        {
            sector
            for res in blind_results.values()
            for sector in res.get("sealed_sectors_touched", [])
        }
    )
    results = {
        "protocol_version": record.protocol_version,
        "cohort": manifest.name,
        "purpose": manifest.purpose,
        "is_discovery": is_discovery,
        "status": status,
        "freeze": {
            "code_sha": record.code_sha,
            "created_utc": record.created_utc,
            "unblinded_utc": record.unblinded_utc,
        },
        "sealed_sectors_touched": sealed,
        "systems": systems_out,
        "blocked_systems": blocked,
        "n_pairs_ranked": n_pairs_ranked,
        "candidates": candidates,
        "reviewed": reviewed,
    }
    if log_path is not None:
        _freeze.log_access(
            log_path,
            {
                "event": "discovery_run",
                "freeze_code_sha": record.code_sha,
                "cohort": manifest.name,
                "is_discovery": is_discovery,
                "status": status,
                "n_candidates": len(candidates),
            },
        )
    return results


def _records_of(blind_result: dict[str, Any]) -> dict[str, EventRecord]:
    return {
        f"k{i}": EventRecord.from_dict(e)
        for i, e in enumerate(blind_result["events"])
    }


def _event_summary(rec: EventRecord) -> dict[str, Any]:
    return {
        "sector": rec.sector,
        "t0": rec.t0,
        "depth": rec.depth,
        "duration_days": rec.duration_days,
        "snr": rec.snr,
    }


def _cross_epoch_pairs(
    records: dict[str, EventRecord],
    windows: dict[int, list[tuple[float, float]]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Early x 106 (or cross-sector) proposal pairs, scored and aliased."""
    from tess_assoc.manifest import ManifestSector, TracerManifest

    early = [rid for rid, r in records.items() if r.sector != DISCOVERY_SECTOR]
    late = [rid for rid, r in records.items() if r.sector == DISCOVERY_SECTOR]
    new_side = late or list(records)
    if not early or not new_side:
        return []
    alias_manifest = TracerManifest(
        name="discovery-alias",
        tic_id=next(iter(records.values())).tic_id,
        epoch_match_tol_days=0.3,
        matcher_thresholds=dict(thresholds),
        sectors=tuple(
            ManifestSector(sector=s, windows=tuple(w))
            for s, w in sorted(windows.items())
        ),
        events=tuple(),
    )
    all_records = list(records.values())
    pairs = []
    for rid_a in early:
        for rid_b in new_side:
            if rid_a == rid_b:
                continue
            if rid_b < rid_a:
                continue  # one direction only: scoring is fully symmetric
            rec_a, rec_b = records[rid_a], records[rid_b]
            if rec_a.sector == rec_b.sector:
                continue
            decision = match(rec_a, rec_b, thresholds)
            if decision.timing_plausible:
                verdicts = filter_aliases(rec_a, rec_b, alias_manifest, all_records)
                retained = [v.period_days for v in verdicts if v.retained]
                aliases_total = len(verdicts)
            else:
                retained, aliases_total = [], 0
            pairs.append(
                {
                    "a": rid_a,
                    "b": rid_b,
                    "event_a": _event_summary(rec_a),
                    "event_b": _event_summary(rec_b),
                    "compatible": decision.compatible,
                    "score": match_score(decision),
                    "morph_corr": decision.morph_corr,
                    "retained_periods": retained,
                    "aliases_total": aliases_total,
                }
            )
    return pairs


def render_discovery_report(results: dict[str, Any]) -> str:
    """Candidate report (candidates are not confirmed planets)."""
    lines = [
        f"# Discovery report: {results['cohort']} (protocol {results['protocol_version']})",
        f"Status: {results['status']}; purpose: {results.get('purpose', 'rehearsal')}; "
        f"discovery data: {results['is_discovery']}.",
        f"Freeze {results['freeze']['code_sha'][:12]}; "
        f"sealed sectors touched: {results['sealed_sectors_touched']}.",
        "",
        "## Systems",
    ]
    for name, system in results["systems"].items():
        if system.get("status") == "blocked-on-archive":
            lines.append(f"- {name}: BLOCKED ON ARCHIVE ({system.get('reason', '')[:80]})")
        elif system.get("status") == "failed":
            lines.append(f"- {name}: FAILED ({system.get('reason', '')[:80]})")
        else:
            lines.append(
                f"- {name}: sectors {system['sectors']}, "
                f"{system['n_proposals']} proposals, "
                f"{system['n_cross_pairs']} cross-epoch pairs, "
                f"anchor pair: {system.get('pair_outcome', 'n/a')}"
            )
    lines += ["", f"## Candidates ({len(results['candidates'])}) — NOT confirmed planets"]
    for cand in results["candidates"]:
        lines.append(
            f"- TIC {cand['tic_id']} ({cand['system']}): "
            f"S{cand['event_a']['sector']}@{cand['event_a']['t0']:.3f} + "
            f"S{cand['event_b']['sector']}@{cand['event_b']['t0']:.3f}, "
            f"score {cand['score']:.3f}, "
            f"{len(cand['retained_periods'])} alias(es) retained"
        )
        lines.append(
            f"  vetting: contamination {cand['vetting']['contamination']['status']}, "
            f"TOI {cand['vetting']['cross_match']['status']}, "
            f"companion {cand['vetting']['companion']['status']}, "
            f"secondaries {cand['vetting']['secondary']['n_flagged']}; "
            f"manual: {', '.join(cand['promotion']['manual_checklist'][:2])} + "
            f"{len(cand['promotion']['manual_checklist']) - 2} more"
        )
    lines += ["", f"## Reviewed, not promoted ({len(results['reviewed'])})"]
    for entry in results["reviewed"][:10]:
        lines.append(
            f"- TIC {entry['tic_id']}: {'; '.join(entry['promotion']['reasons'])}"
        )
    if len(results["reviewed"]) > 10:
        lines.append(f"- ... and {len(results['reviewed']) - 10} more")
    return "\n".join(lines) + "\n"


__all__ = [
    "DISCOVERY_SECTOR",
    "DiscoveryManifest",
    "DiscoverySystem",
    "cone_tics",
    "load_discovery_manifest",
    "render_discovery_report",
    "run_discovery",
    "sectors_for_tic",
    "select_cohort",
]
