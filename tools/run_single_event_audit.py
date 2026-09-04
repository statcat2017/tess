"""Run the reproducible audit for the current isolated-event queue."""

from __future__ import annotations

import json
import math
from pathlib import Path

from tess_assoc.audit import build_single_event_audit, render_single_event_audit
from tess_assoc.discovery import sectors_for_tic
from tess_assoc.vetting import (
    check_companion_radius,
    check_contamination,
    check_variables,
    cross_match_ctoi,
    cross_match_toi,
    stellar_radius,
    tic_coords,
)


CACHE = Path("/tmp/long_followup_cache")
CASES = (
    {
        "tic_id": 137801807,
        "sector": 69,
        "t0": 3204.418352997779,
        "depth": 0.05199308885584297,
        "duration_days": 0.12731087503289018,
        "snr": 30.817230454719876,
    },
    {
        "tic_id": 117549174,
        "sector": 69,
        "t0": 3183.133477153529,
        "depth": 0.03180105985232673,
        "duration_days": 0.06944716061070721,
        "snr": 11.671593198637387,
    },
)


def _gaia_neighbors(ra: float, dec: float) -> list[dict]:
    from astroquery.gaia import Gaia

    query = (
        "SELECT TOP 50 source_id,ra,dec,phot_g_mean_mag,parallax "
        "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec), "
        f"CIRCLE('ICRS',{ra},{dec},60/3600.0)) ORDER BY phot_g_mean_mag"
    )
    rows = Gaia.launch_job(query).get_results()
    out = []
    cos_dec = math.cos(math.radians(dec))
    for row in rows:
        row_ra, row_dec = float(row["ra"]), float(row["dec"])
        separation = math.hypot(
            (row_ra - ra) * cos_dec, row_dec - dec
        ) * 3600.0
        g_mag = row["phot_g_mean_mag"]
        if not math.isfinite(float(g_mag)):
            continue
        out.append(
            {
                "source_id": int(row["source_id"]),
                "separation_arcsec": separation,
                "g_mag": float(g_mag),
                "parallax_mas": float(row["parallax"]),
            }
        )
    return out


def main() -> None:
    reports = []
    for event in CASES:
        tic_id = event["tic_id"]
        coords = tic_coords(tic_id)
        if coords is None:
            raise RuntimeError(f"no TIC coordinates for {tic_id}")
        from astroquery.mast import Catalogs

        row = Catalogs.query_criteria(catalog="Tic", ID=tic_id)[0]
        tmag = float(row["Tmag"])
        rad = stellar_radius(tic_id)
        catalogs = {
            "contamination": check_contamination(tic_id),
            "cross_match": cross_match_toi(tic_id),
            "ctoi": cross_match_ctoi(tic_id),
            "variables": check_variables(tic_id),
            "companion": check_companion_radius(tic_id, event["depth"], rad=rad),
        }
        path = CACHE / (
            f"hlsp_tess-spoc_tess_phot_{tic_id:016d}-s{event['sector']:04d}_tess_v1_lc.fits"
        )
        report = build_single_event_audit(
            tic_id,
            event,
            [{"sector": event["sector"], "local_path": str(path)}],
            metadata={
                "tmag": tmag,
                "teff": float(row["Teff"]),
                "stellar_radius_r_sun": rad,
                "available_sectors": sectors_for_tic(tic_id),
            },
            catalogs=catalogs,
            neighbors=_gaia_neighbors(*coords),
        )
        reports.append(report)
    output = Path("reports")
    output.mkdir(exist_ok=True)
    (output / "single_event_audit.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    (output / "single_event_audit.html").write_text(render_single_event_audit(reports))
    for report in reports:
        print(
            report["tic_id"],
            report["checks"],
            report["event_product"]["flux"]["channels"],
        )
    print(f"wrote {output}/single_event_audit.json and .html")


if __name__ == "__main__":
    main()
