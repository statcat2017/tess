"""Run neighbour, flagged-cadence, sensitivity, and MAST inventory checks."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.archive import ArchiveProduct
from tess_assoc.audit import measure_flux_channel
from tess_assoc.extract import load_lightcurve
from tess_assoc.inject import inject_transit
from tess_assoc.pixel_audit import flagged_event_diagnostic
from tess_assoc.propose import propose_events, records_from_proposals


CACHE = Path("/tmp/long_followup_cache")
TESSCUT_CACHE = Path("/tmp/tesscut_cache")
CASES = {
    137801807: {
        "sector": 69,
        "t0": 3204.418352997779,
        "duration_days": 0.12731087503289018,
    },
    117549174: {
        "sector": 69,
        "t0": 3183.133477153529,
        "duration_days": 0.06944716061070721,
    },
}


def _product(tic_id: int, sector: int) -> ArchiveProduct:
    path = CACHE / (
        f"hlsp_tess-spoc_tess_phot_{tic_id:016d}-s{sector:04d}_tess_v1_lc.fits"
    )
    return ArchiveProduct(
        tic_id=tic_id,
        sector=sector,
        local_path=str(path),
        data_uri="",
        retrieved_utc="",
        cached=True,
    )


def _separation(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    return math.hypot(
        (ra_a - ra_b) * math.cos(math.radians(dec_a)), dec_a - dec_b
    ) * 3600.0


def _gaia_neighbours(ra: float, dec: float) -> list[dict]:
    from astroquery.gaia import Gaia

    query = (
        "SELECT TOP 50 source_id,ra,dec,phot_g_mean_mag,parallax "
        "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec), "
        f"CIRCLE('ICRS',{ra},{dec},60/3600.0))"
    )
    rows = Gaia.launch_job(query).get_results()
    out = []
    for row in rows:
        row_ra, row_dec = float(row["ra"]), float(row["dec"])
        separation = _separation(ra, dec, row_ra, row_dec)
        if not 0.5 <= separation <= 10.0:
            continue
        g_mag = float(row["phot_g_mean_mag"])
        if math.isfinite(g_mag):
            out.append(
                {
                    "source_id": int(row["source_id"]),
                    "ra_deg": row_ra,
                    "dec_deg": row_dec,
                    "separation_arcsec": separation,
                    "g_mag": g_mag,
                    "parallax_mas": float(row["parallax"]),
                }
            )
    return sorted(out, key=lambda item: item["separation_arcsec"])


def _tic_neighbour(ra: float, dec: float, *, exclude_tic_id: int) -> dict:
    from astroquery.mast import Catalogs

    rows = Catalogs.query_region(f"{ra} {dec}", radius="0.001 deg", catalog="Tic")
    candidates = []
    for row in rows:
        try:
            row_ra, row_dec = float(row["ra"]), float(row["dec"])
            tic_id = int(row["ID"])
            if tic_id == exclude_tic_id:
                continue
            candidates.append(
                {
                    "tic_id": tic_id,
                    "separation_arcsec": _separation(ra, dec, row_ra, row_dec),
                    "tmag": float(row["Tmag"]),
                    "stellar_radius_r_sun": float(row["rad"]),
                }
            )
        except (TypeError, ValueError, AttributeError):
            continue
    if not candidates:
        return {"status": "no-tic-match"}
    best = min(candidates, key=lambda item: item["separation_arcsec"])
    status = "matched" if best["separation_arcsec"] <= 1.5 else "offset-match-review"
    return {"status": status, **best}


def _neighbour_attribution(pixel_reports: list[dict]) -> list[dict]:
    output = []
    for report in pixel_reports:
        tic_id = report["tic_id"]
        event = report["event"]
        if tic_id not in CASES:
            continue
        neighbours = []
        for neighbour in report.get("gaia_neighbours", []):
            entry = dict(neighbour)
            entry["tic"] = _tic_neighbour(
                neighbour["ra_deg"], neighbour["dec_deg"], exclude_tic_id=tic_id
            )
            neighbour_path = TESSCUT_CACHE / (
                f"tesscut_tic{tic_id}_s{event['sector']:04d}.fits"
            )
            if neighbour_path.exists():
                try:
                    entry["neighbour_centred_audit"] = flagged_event_diagnostic(
                        neighbour_path,
                        ra_deg=neighbour["ra_deg"],
                        dec_deg=neighbour["dec_deg"],
                        sector=event["sector"],
                        t0=event["t0"],
                        duration_days=event["duration_days"],
                    )
                except Exception as error:  # noqa: BLE001 — preserve partial result
                    entry["audit_error"] = str(error)[:200]
            neighbours.append(entry)
        output.append({"tic_id": tic_id, "neighbours": neighbours})
    return output


def _best_injection_center(time: list[float], half_span: float = 0.6) -> float:
    lo, hi = time[0] + half_span, time[-1] - half_span
    if hi <= lo:
        return time[len(time) // 2]
    candidates = [lo + (hi - lo) * index / 32 for index in range(33)]
    return max(
        candidates,
        key=lambda center: sum(abs(value - center) <= half_span for value in time),
    )


def _injection_centers(time: list[float]) -> list[float]:
    lo, hi = time[0] + 0.8, time[-1] - 0.8
    raw = [lo + (hi - lo) * index / 4 for index in range(5)]
    centers = []
    for guess in raw:
        local = [value for value in time if abs(value - guess) <= 0.4]
        centers.append(_best_injection_center(local or time))
    return centers


def _sensitivity() -> list[dict]:
    rows = []
    for sector in (2, 29, 70):
        path = _product(117549174, sector)
        time, flux = load_lightcurve(path)
        curve = {
            "sector": sector,
            "centers": _injection_centers(time),
            "cadences": len(time),
            "measurements": [],
        }
        for center in curve["centers"]:
            for depth in (0.005, 0.01, 0.02, 0.0318, 0.05):
                injected = inject_transit(
                    time, flux, center, CASES[117549174]["duration_days"], depth
                )
                proposals = propose_events(time, injected)
                records, skipped = records_from_proposals(
                    time,
                    injected,
                    proposals,
                    tic_id=117549174,
                    sector=sector,
                    quality_base={"role": "candidate-sensitivity-injection"},
                )
                detected = any(abs(record.t0 - center) <= 0.3 for record in records.values())
                measured = measure_flux_channel(
                    time, injected, center, CASES[117549174]["duration_days"]
                )
                curve["measurements"].append(
                    {
                        "center": center,
                        "depth_injected": depth,
                        "proposals": len(proposals),
                        "events_extracted": len(records),
                        "detected_at_injection": detected,
                        "skipped": [entry.reason for entry in skipped],
                        "measured": measured,
                    }
                )
        rows.append(curve)
    return rows


def _mast_inventory() -> list[dict]:
    from astroquery.mast import Observations

    results = []
    for tic_id, case in CASES.items():
        rows = Observations.query_criteria(
            target_name=str(tic_id), obs_collection="HLSP", dataproduct_type="timeseries"
        )
        products = []
        for row in rows:
            products.append(
                {
                    "provider": str(row["provenance_name"]),
                    "sector": int(row["sequence_number"]),
                    "obs_id": str(row["obs_id"]),
                }
            )
        results.append(
            {
                "tic_id": tic_id,
                "anchor_sector": case["sector"],
                "products": sorted(products, key=lambda item: (item["sector"], item["provider"])),
            }
        )
    return results


def _render(results: dict) -> str:
    lines = [
        "# Candidate extension findings",
        f"Run: {results['run_utc']}",
        "",
        "This report covers neighbour attribution, quality-flag diagnostics, "
        "injection sensitivity, and public MAST product inventory. It does not "
        "run the real Survey B cohort.",
        "",
        "## Neighbour attribution",
    ]
    for target in results["neighbour_attribution"]:
        lines.append(f"### TIC {target['tic_id']}")
        for neighbour in target["neighbours"]:
            tic = neighbour["tic"]
            lines.append(
                f"- Gaia {neighbour['source_id']} at {neighbour['separation_arcsec']:.2f}\"; "
                f"TIC lookup: {tic.get('status')} {tic.get('tic_id', '')}."
            )
    flagged = results["flagged_tic117549174"]
    apertures = flagged["all_cadence_diagnostic"]["apertures"]
    depths = ", ".join(f"{row.get('depth', float('nan')):.4f}" for row in apertures)
    lines.extend(
        [
            "",
            "## TIC 117549174 flagged-cadence diagnostic",
            f"- {flagged['event_quality_flagged']}/{flagged['event_cadences']} event cadences are flagged "
            f"with QUALITY values {flagged['event_quality_values']}; valid-cadence audit remains insufficient.",
            f"- All-cadence target-centred 1/3/5-pixel depth values: {depths}; "
            "negative values are brightenings, not transit-like dimmings.",
            "- The same all-cadence aperture values are returned at both Gaia comparison positions, "
            "so this diagnostic is spatially broad and not usable localisation evidence.",
        ]
    )
    lines.extend(["", "## TIC 117549174 sensitivity"])
    for sector in results["sensitivity"]:
        by_depth = {}
        for row in sector["measurements"]:
            by_depth.setdefault(row["depth_injected"], []).append(row["detected_at_injection"])
        values = ", ".join(
            f"{depth:.4f}:{sum(found)}/{len(found)}"
            for depth, found in sorted(by_depth.items())
        )
        lines.append(
            f"- Sector {sector['sector']} at {len(sector['centers'])} placements: {values}"
        )
    lines.extend(["", "## MAST provider inventory"])
    for target in results["mast_inventory"]:
        providers = sorted({item["provider"] for item in target["products"]})
        anchor = sorted({item["provider"] for item in target["products"] if item["sector"] == target["anchor_sector"]})
        lines.append(
            f"- TIC {target['tic_id']}: providers {', '.join(providers)}; "
            f"anchor-sector providers {', '.join(anchor) or 'none'}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    pixel_reports = json.loads(Path("reports/tesscut_pixel_audit.json").read_text())
    tic117_pixel = next(report for report in pixel_reports if report["tic_id"] == 117549174)
    comparison_positions = {
        label: (
            value["ra_deg"],
            value["dec_deg"],
        )
        for label, value in {
            f"gaia-neighbour-{index}": row
            for index, row in enumerate(tic117_pixel.get("gaia_neighbours", []), start=1)
        }.items()
    }
    event = tic117_pixel["event"]
    flagged = flagged_event_diagnostic(
        TESSCUT_CACHE / tic117_pixel["pixel"]["path"],
        ra_deg=event["ra_deg"],
        dec_deg=event["dec_deg"],
        sector=event["sector"],
        t0=event["t0"],
        duration_days=event["duration_days"],
        comparison_positions=comparison_positions,
    )
    results = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "neighbour_attribution": _neighbour_attribution(pixel_reports),
        "flagged_tic117549174": flagged,
        "sensitivity": _sensitivity(),
        "mast_inventory": _mast_inventory(),
    }
    output = Path("reports")
    output.mkdir(exist_ok=True)
    (output / "candidate_extension_findings.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (output / "candidate_extension_findings.md").write_text(_render(results))
    print(_render(results))
    print("wrote reports/candidate_extension_findings.json and .md")


if __name__ == "__main__":
    main()
