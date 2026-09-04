"""Fetch Sector 69 TESSCut data and publish pixel-level audits."""

from __future__ import annotations

import json
from pathlib import Path

from tess_assoc.pixel_audit import (
    audit_tesscut_file,
    download_tesscut,
    render_pixel_audit_html,
)


CASES = (
    {
        "tic_id": 137801807,
        "ra_deg": 344.099833723843,
        "dec_deg": -20.2794162934567,
        "sector": 69,
        "t0": 3204.418352997779,
        "depth": 0.05199308885584297,
        "duration_days": 0.12731087503289018,
        "snr": 30.817230454719876,
    },
    {
        "tic_id": 117549174,
        "ra_deg": 0.142169340507106,
        "dec_deg": -13.4153343283066,
        "sector": 69,
        "t0": 3183.133477153529,
        "depth": 0.03180105985232673,
        "duration_days": 0.06944716061070721,
        "snr": 11.671593198637387,
    },
)


def main() -> None:
    reports = []
    for event in CASES:
        path = download_tesscut(
            event["ra_deg"],
            event["dec_deg"],
            event["sector"],
            size=9,
            directory="/tmp/tesscut_cache",
            name=f"tesscut_tic{event['tic_id']}_s{event['sector']:04d}.fits",
        )
        reports.append(
            {
                "tic_id": event["tic_id"],
                "event": event,
                "pixel": audit_tesscut_file(
                    path,
                    ra_deg=event["ra_deg"],
                    dec_deg=event["dec_deg"],
                    sector=event["sector"],
                    t0=event["t0"],
                    duration_days=event["duration_days"],
                ),
            }
        )
    output = Path("reports")
    output.mkdir(exist_ok=True)
    (output / "tesscut_pixel_audit.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    (output / "tesscut_pixel_audit.html").write_text(render_pixel_audit_html(reports))
    print(f"wrote {output}/tesscut_pixel_audit.json and .html")


if __name__ == "__main__":
    main()
