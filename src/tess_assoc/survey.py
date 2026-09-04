"""Archive-mining survey at scale (issue #9).

Cohort enumeration (TOI-seeded or cone-seeded), threaded harvesting with
per-star fault isolation and resume, batched known-planet cross-matching,
and a global triage. Designed to stream: per-system harvests persist as
JSON lines, so interrupted runs resume instead of restarting.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from tess_assoc import freeze as _freeze
from tess_assoc._validate import require_positive_finite, require_strict_int
from tess_assoc.archive import ArchiveUnavailable
from tess_assoc.discovery import (
    harvest_system,
    load_discovery_manifest,
    render_discovery_report,
    triage_ranked_pairs,
)
from tess_assoc.vetting import cross_match_tois


def fetch_tois_box(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    *,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """TOIs in an RA/Dec box with ephemeris columns (one TAP call)."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
        NasaExoplanetArchive,
    )

    try:
        tab = NasaExoplanetArchive.query_criteria(
            table="toi",
            where=f"ra>{ra_min} and ra<{ra_max} and dec>{dec_min} and dec<{dec_max}",
            select=(
                f"top {limit} tid,toi,ra,dec,pl_orbper,"
                "pl_tranmid,pl_trandurh,tfopwg_disp"
            ),
        )
    except Exception as e:
        raise ArchiveUnavailable(f"TOI box query failed: {e}") from e
    out = []
    for r in tab:
        try:
            out.append(
                {
                    "tic_id": int(r["tid"]),
                    "toi": str(r["toi"]),
                    "disposition": str(r["tfopwg_disp"]),
                    "period_days": float(r["pl_orbper"].value),
                    "t0_bjd_tdb": float(r["pl_tranmid"]),
                    "duration_hours": float(r["pl_trandurh"].value),
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def build_survey_manifest(
    name: str,
    targets: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    purpose: str = "mining",
    ephemeris_source: str = "survey assembly",
    epoch_match_tol_days: float = 0.3,
    window_half_span_days: float = 0.6,
    resample_samples: int = 61,
) -> dict[str, Any]:
    """Manifest JSON from target dicts (pure assembly, offline-testable).

    Each target needs tic_id, sectors, name; ephemeris keys optional.
    """
    if purpose not in ("rehearsal", "mining", "discovery"):
        raise ValueError("purpose must be rehearsal, mining, or discovery")
    systems = []
    for target in targets:
        require_strict_int("tic_id", target["tic_id"], minimum=1)
        sectors = target.get("sectors")
        if not isinstance(sectors, (list, tuple)) or not sectors:
            raise ValueError(f"target {target['tic_id']} needs non-empty sectors")
        system: dict[str, Any] = {
            "name": target.get("name", f"TIC {target['tic_id']}"),
            "tic_id": target["tic_id"],
            "sectors": list(sectors),
        }
        for key in ("toi", "period_days", "t0_bjd_tdb", "duration_hours"):
            if target.get(key) is not None:
                system[key] = target[key]
        systems.append(system)
    if not systems:
        raise ValueError("survey needs at least one target system")
    return {
        "name": name,
        "product": "TESS-SPOC FFI",
        "ephemeris_source": ephemeris_source,
        "epoch_match_tol_days": epoch_match_tol_days,
        "window_half_span_days": window_half_span_days,
        "resample_samples": resample_samples,
        "matcher_thresholds": dict(thresholds),
        "purpose": purpose,
        "systems": systems,
    }


def resolve_coverage(
    tic_ids: list[int], *, max_workers: int = 8
) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Threaded SPOC sector lookup; per-TIC faults isolated, never fatal."""
    from tess_assoc.discovery import sectors_for_tic

    coverage: dict[int, list[int]] = {}
    failures: dict[int, str] = {}

    def lookup(tic_id: int) -> tuple[int, list[int] | None, str | None]:
        try:
            return tic_id, sectors_for_tic(tic_id), None
        except Exception as e:  # noqa: BLE001 — fault isolation is the point
            return tic_id, None, str(e)[:200]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for tic_id, sectors, error in pool.map(lookup, list(dict.fromkeys(tic_ids))):
            if error is None:
                coverage[tic_id] = sectors
            else:
                failures[tic_id] = error
    return coverage, failures


def _read_state(state_path: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    path = Path(state_path)
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            done[entry["system"]] = entry
    return done


def run_mining_survey(
    manifest_path: str,
    *,
    freeze_path: str,
    config,
    cache_dir: str | None = None,
    out_dir: str,
    shortlist_k: int = 10,
    max_workers: int = 4,
    log_path: str | None = None,
) -> dict[str, Any]:
    """Threaded survey: harvest per star (resume), triage globally once."""
    require_strict_int("shortlist_k", shortlist_k, minimum=1)
    require_positive_finite("max_workers", max_workers)
    manifest = load_discovery_manifest(manifest_path, freeze_path, config)
    record = _freeze.load_freeze_record(freeze_path)
    record = _freeze.mark_unblinded(freeze_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = str(out / "harvest.jsonl")
    done = _read_state(state_path)

    todo = [s for s in manifest.systems if s.name not in done]
    harvests: dict[str, dict[str, Any]] = {
        name: {
            "status": entry["status"],
            "systems_out": entry["systems_out"],
            "products": entry.get("products", []),
            "pairs": entry.get("pairs", []),
        }
        for name, entry in done.items()
    }

    def harvest_one(system) -> tuple[str, dict[str, Any]]:
        try:
            harvest = harvest_system(
                manifest, system, record=record, cache_dir=cache_dir
            )
        except Exception as e:  # noqa: BLE001 — one bad star never kills a survey
            import traceback

            return system.name, {
                "status": "failed",
                "systems_out": {
                    "tic_id": system.tic_id,
                    "sectors": list(system.sectors),
                    "status": "failed",
                    "reason": str(e)[:300],
                    "traceback": traceback.format_exc(limit=3),
                },
                "products": [],
                "pairs": [],
            }
        thin = {
            "status": harvest["status"],
            "systems_out": harvest["systems_out"],
            "products": harvest.get("products", []),
            "pairs": harvest["pairs"],
        }
        return system.name, thin

    def append_state(name: str, thin: dict[str, Any]) -> None:
        with open(state_path, "a") as f:
            f.write(json.dumps({"system": name, **thin}) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(harvest_one, s): s for s in todo}
        for future in concurrent.futures.as_completed(futures):
            name, thin = future.result()
            harvests[name] = thin
            append_state(name, thin)

    for name, harvest in harvests.items():
        if harvest["status"] not in ("complete", "blocked-on-archive", "failed"):
            raise ValueError(f"corrupt harvest state for {name}")

    triage_in = {
        name: h for name, h in harvests.items() if h["status"] == "complete"
    }
    shortlist_tics = _shortlist_tics(manifest, triage_in, shortlist_k)
    prefetch = _prefetch_catalog(manifest, shortlist_tics)
    candidates, reviewed = triage_ranked_pairs(
        manifest, triage_in, shortlist_k=shortlist_k, catalog_prefetch=prefetch
    )
    n_pairs_ranked = sum(len(h["pairs"]) for h in triage_in.values())
    blocked = sorted(
        name for name, h in harvests.items() if h["status"] == "blocked-on-archive"
    )
    failed = sorted(
        name for name, h in harvests.items() if h["status"] == "failed"
    )
    is_discovery = any(
        106 in s.sectors for s in manifest.systems
    )
    if blocked and not triage_in:
        status = "blocked-on-archive"
    elif blocked or failed:
        status = "partial"
    else:
        status = "complete"
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
        "sealed_sectors_touched": [],
        "systems": {name: h["systems_out"] for name, h in harvests.items()},
        "blocked_systems": blocked,
        "failed_systems": failed,
        "n_pairs_ranked": n_pairs_ranked,
        "candidates": candidates,
        "reviewed": reviewed,
    }
    (out / "survey_results.json").write_text(json.dumps(results) + "\n")
    if log_path is not None:
        _freeze.log_access(
            log_path,
            {
                "event": "mining_survey",
                "freeze_code_sha": record.code_sha,
                "cohort": manifest.name,
                "status": status,
                "n_systems": len(harvests),
                "n_candidates": len(candidates),
            },
        )
    return results


def _shortlist_tics(manifest, triage_in, shortlist_k: int) -> list[int]:
    scored = sorted(
        (
            (pair["score"], system_tic(manifest, name))
            for name, harvest in triage_in.items()
            for pair in harvest["pairs"]
        ),
        reverse=True,
    )
    return [tic for _, tic in scored[:shortlist_k]]


def system_tic(manifest, name: str) -> int:
    return next(s.tic_id for s in manifest.systems if s.name == name)


def _prefetch_catalog(manifest, tic_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Batched TOI cross-match for shortlisted TICs (one TAP call)."""
    from tess_assoc.vetting import (
        check_contamination,
        check_variables,
        cross_match_ctoi,
        cross_match_tois,
        stellar_radius,
    )

    batch = cross_match_tois(tic_ids)
    prefetch: dict[int, dict[str, Any]] = {}
    for tic in dict.fromkeys(tic_ids):
        if batch["ok"] and tic in batch["matches"]:
            rows = batch["matches"][tic]
            cross_match = (
                {"status": "clean", "matches": [], "reason": None}
                if not rows
                else {"status": "known-toi", "matches": rows, "reason": None}
            )
        else:
            cross_match = {
                "status": "unknown",
                "matches": [],
                "reason": batch.get("reason") or "not in batch result",
            }
        try:
            contamination = check_contamination(tic)
        except Exception as e:  # noqa: BLE001 — fail open into manual review
            contamination = {"status": "unknown", "reason": str(e)[:200]}
        prefetch[tic] = {
            "contamination": contamination,
            "cross_match": cross_match,
            "stellar_rad": stellar_radius(tic),
            "variables": check_variables(tic),
            "ctoi": cross_match_ctoi(tic),
        }
    return prefetch


def render_survey_report(results: dict[str, Any]) -> str:
    """Short survey wrapper around the discovery report."""
    from tess_assoc.discovery import render_discovery_report

    header = (
        f"Survey: {len(results['systems'])} systems, "
        f"{results['n_pairs_ranked']} pairs ranked, "
        f"{len(results['candidates'])} candidates, "
        f"{len(results['reviewed'])} reviewed."
    )
    if results.get("failed_systems"):
        header += f" Failed: {', '.join(results['failed_systems'])}."
    return header + "\n\n" + render_discovery_report(results)


__all__ = [
    "build_survey_manifest",
    "fetch_tois_box",
    "render_survey_report",
    "resolve_coverage",
    "run_mining_survey",
    "system_tic",
]
