"""Run the bounded real Survey B eclipsing-binary cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.archive import ArchiveProduct
from tess_assoc.bulk import bulk_fetch
from tess_assoc.circumbinary import (
    BinaryPilotTarget,
    eclipse_windows,
    render_binary_pilot_report,
    run_binary_pilot,
)
from tess_assoc.observability import coverage_windows
from tess_assoc.extract import BTJD_OFFSET, load_lightcurve
from tess_assoc.propose import propose_events, records_from_proposals


CATALOGUE_URL = (
    "https://raw.githubusercontent.com/bdrdavies/mono-cbp/"
    "8d373b48917eb97f9ff18378184581943697fe6a/catalogues/TEBC_morph_05_P_7.csv"
)
SECTOR_TIMES_URL = (
    "https://raw.githubusercontent.com/bdrdavies/mono-cbp/"
    "8d373b48917eb97f9ff18378184581943697fe6a/catalogues/sector_times.csv"
)
CATALOGUE_CACHE = Path("/tmp/survey_b/TEBC_morph_05_P_7.csv")
DATA_CACHE = Path("/tmp/survey_b/spoc")
DEV_MAX_SECTOR = 79


def _download_if_needed(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tess-assoc/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value or value.lower() in {"nan", "none"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _sector_list(value: str) -> tuple[int, ...]:
    sectors = []
    for raw in value.split(","):
        try:
            sector = int(raw.strip())
        except ValueError:
            continue
        if 1 <= sector <= DEV_MAX_SECTOR:
            sectors.append(sector)
    return tuple(sorted(set(sectors)))


def load_tebc_targets(path: Path) -> tuple[tuple[BinaryPilotTarget, ...], dict[str, int]]:
    """Convert the published TEBC rows into the Survey B target contract."""
    targets: dict[int, BinaryPilotTarget] = {}
    excluded = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                tic_id = int(row["tess_id"])
            except (KeyError, ValueError):
                excluded["invalid-tic"] += 1
                continue
            if tic_id in targets:
                excluded["duplicate-tic"] += 1
                continue
            period = _number(row, "period")
            bjd0 = _number(row, "bjd0")
            primary_width = _number(row, "prim_width_2g")
            secondary_width = _number(row, "sec_width_2g")
            secondary_phase = _number(row, "sec_pos_2g")
            sectors = _sector_list(row.get("sectors", ""))
            if period is None or bjd0 is None:
                excluded["missing-ephemeris"] += 1
                continue
            if primary_width is None or primary_width <= 0:
                excluded["missing-primary-width"] += 1
                continue
            if secondary_width is None or secondary_width <= 0:
                excluded["missing-secondary-width"] += 1
                continue
            if secondary_phase is None or not 0 < secondary_phase < 1:
                excluded["missing-secondary-phase"] += 1
                continue
            if not sectors:
                excluded["no-development-sector"] += 1
                continue
            targets[tic_id] = BinaryPilotTarget(
                tic_id=tic_id,
                name=f"TEBC-{tic_id}",
                period_days=period,
                # TEBC calls this column bjd0, but its values are BTJD-scale
                # (for example ~1553), unlike the BJD contract used here.
                t0_bjd_tdb=bjd0 + BTJD_OFFSET,
                primary_duration_days=period * primary_width,
                secondary_phase=secondary_phase,
                secondary_duration_days=period * secondary_width,
                sectors=sectors,
                target_class="eclipsing-binary-control",
            )
    return tuple(targets.values()), dict(excluded)


def _product(tic_id: int, sector: int, path: str) -> ArchiveProduct:
    return ArchiveProduct(
        tic_id=tic_id,
        sector=sector,
        local_path=path,
        data_uri="",
        retrieved_utc="",
        cached=True,
    )


def _process_target(
    target: BinaryPilotTarget,
    files: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[Any], dict[int, list[tuple[float, float]]], dict[str, Any]]:
    events: list[Any] = []
    coverage: dict[int, list[tuple[float, float]]] = {}
    sectors: dict[str, Any] = {}
    for sector in target.sectors:
        result = files.get((target.tic_id, sector))
        sector_result: dict[str, Any] = {"status": "missing"}
        if not result or result["status"] not in {"cached", "downloaded"}:
            if result:
                sector_result.update(result)
            sectors[str(sector)] = sector_result
            continue
        try:
            product = _product(target.tic_id, sector, result["local_path"])
            time, flux = load_lightcurve(product)
            spans = coverage_windows(time)
            coverage[sector] = spans
            masks = eclipse_windows(target, {sector: spans}).get(sector, [])
            keep = [
                not any(start <= point <= end for start, end in masks)
                for point in time
            ]
            search_time = [point for point, use in zip(time, keep) if use]
            search_flux = [value for value, use in zip(flux, keep) if use]
            proposals = propose_events(search_time, search_flux)
            records, skipped = records_from_proposals(
                search_time,
                search_flux,
                proposals,
                tic_id=target.tic_id,
                sector=sector,
                quality_base={
                    "role": "survey-b-real-proposal",
                    "source_catalog": "TEBC_morph_05_P_7",
                    "binary_eclipses_masked": True,
                },
            )
            events.extend(records.values())
            sector_result = {
                "status": "complete",
                "cadences": len(time),
                "masked_cadences": len(time) - len(search_time),
                "coverage_windows": spans,
                "n_proposals": len(proposals),
                "n_events_extracted": len(records),
                "n_skipped": len(skipped),
                "skipped_reasons": Counter(item.reason for item in skipped),
            }
        except Exception as error:  # noqa: BLE001 — one archive file cannot stop cohort
            sector_result = {"status": "failed", "reason": str(error)[:300]}
        sectors[str(sector)] = sector_result
    return events, coverage, sectors


def _json_safe(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(output_dir: str = "reports") -> dict[str, Any]:
    _download_if_needed(CATALOGUE_URL, CATALOGUE_CACHE)
    targets, excluded = load_tebc_targets(CATALOGUE_CACHE)
    pairs = [(target.tic_id, sector) for target in targets for sector in target.sectors]
    fetched = bulk_fetch(pairs, str(DATA_CACHE), max_workers=8, timeout=120)
    file_results = {
        (int(item["tic_id"]), int(item["sector"])): item
        for bucket in fetched.values()
        for item in bucket
    }

    all_events: list[Any] = []
    all_coverage: dict[int, list[tuple[float, float]]] = {}
    per_target: dict[str, Any] = {}
    for target in targets:
        events, coverage, sector_results = _process_target(target, file_results)
        all_events.extend(events)
        all_coverage.update(coverage)
        per_target[target.name] = {
            "tic_id": target.tic_id,
            "sectors": list(target.sectors),
            "period_days": target.period_days,
            "n_events": len(events),
            "sectors_run": sector_results,
        }

    result = run_binary_pilot(targets, all_events, all_coverage)
    result["run_utc"] = datetime.now(timezone.utc).isoformat()
    result["source"] = {
        "catalogue_url": CATALOGUE_URL,
        "catalogue_sha256": _sha256(CATALOGUE_CACHE),
        "sector_times_url": SECTOR_TIMES_URL,
        "development_sector_max": DEV_MAX_SECTOR,
        "catalogue_epoch_scale": "TEBC bjd0 interpreted as BTJD; converted to BJD_TDB by adding 2457000",
        "catalogue_cache": str(CATALOGUE_CACHE),
        "data_cache": str(DATA_CACHE),
    }
    result["input_summary"] = {
        "catalogue_rows": sum(1 for _ in csv.DictReader(CATALOGUE_CACHE.open())),
        "targets_used": len(targets),
        "excluded_rows": excluded,
        "download_status": {key: len(value) for key, value in fetched.items()},
        "target_sector_pairs": len(pairs),
        "events_proposed_and_extracted": len(all_events),
    }
    result["per_target"] = per_target

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_result = _json_safe(result)
    (output / "survey_b_real.json").write_text(json.dumps(safe_result, indent=2) + "\n")
    report = render_binary_pilot_report(safe_result)
    report += "\n## Input summary\n"
    for key, value in safe_result["input_summary"].items():
        report += f"- {key}: {value}\n"
    report += "\nThis is an unlabeled real eclipsing-binary mining cohort. No target is treated as a known circumbinary-planet positive.\n"
    (output / "survey_b_real.md").write_text(report)
    print(report)
    return safe_result


if __name__ == "__main__":
    run()
