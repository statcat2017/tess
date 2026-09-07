"""Search every available SPOC sector for repeats of two isolated events."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.bulk import bulk_fetch
from tess_assoc.candidate_followup import event_summary, rank_repeat_events
from tess_assoc.discovery import sectors_for_tic
from tess_assoc.archive import ArchiveProduct
from tess_assoc.extract import SkippedTransit, extract_at, load_lightcurve
from tess_assoc.matcher import REQUIRED_THRESHOLDS
from tess_assoc.propose import propose_with_detail, records_from_proposals


CACHE = Path("/tmp/long_followup_cache")
THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}
CASES = (
    {
        "tic_id": 137801807,
        "sector": 69,
        "t0": 3204.418352997779,
        "depth": 0.05199308885584297,
        "duration_days": 0.12731087503289018,
        "snr": 30.817593198637387,
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


def _product(tic_id: int, entry: dict) -> ArchiveProduct:
    return ArchiveProduct(
        tic_id=tic_id,
        sector=int(entry["sector"]),
        local_path=entry["local_path"],
        data_uri=entry.get("data_uri", ""),
        retrieved_utc=entry.get("retrieved_utc", ""),
        cached=entry.get("status") == "cached",
    )


def _search_case(case: dict) -> dict:
    tic_id = case["tic_id"]
    try:
        sectors = sectors_for_tic(tic_id)
        coverage_error = None
    except Exception as error:  # noqa: BLE001 — preserve a usable report
        sectors = [2, 29, 69] if tic_id == 137801807 else [2, 29, 69, 70]
        coverage_error = str(error)[:200]
    jobs = [(tic_id, sector) for sector in sectors]
    fetched = bulk_fetch(jobs, str(CACHE), max_workers=4)
    products = {
        entry["sector"]: entry
        for bucket in fetched.values()
        for entry in bucket
        if entry.get("status") in {"cached", "downloaded"}
    }
    anchor_product = products.get(case["sector"])
    if anchor_product is None:
        return {
            "tic_id": tic_id,
            "available_sectors": sectors,
            "coverage_error": coverage_error,
            "status": "blocked-on-archive",
            "fetched": fetched,
        }

    anchor_time, anchor_flux = load_lightcurve(_product(tic_id, anchor_product))
    anchor_result = extract_at(
        anchor_time,
        anchor_flux,
        case["t0"],
        case["duration_days"],
        tic_id=tic_id,
        sector=case["sector"],
        quality={"role": "isolated-event-anchor", "source": "single_event_audit"},
    )
    if isinstance(anchor_result, SkippedTransit):
        return {
            "tic_id": tic_id,
            "available_sectors": sectors,
            "coverage_error": coverage_error,
            "status": "anchor-unmeasurable",
            "reason": anchor_result.reason,
            "fetched": fetched,
        }

    candidate_events = []
    sector_results = []
    for sector in sectors:
        if sector == case["sector"]:
            continue
        product = products.get(sector)
        if product is None:
            sector_results.append({"sector": sector, "status": "unavailable"})
            continue
        try:
            time, flux = load_lightcurve(_product(tic_id, product))
            proposals, _, _ = propose_with_detail(time, flux)
            records, skipped = records_from_proposals(
                time,
                flux,
                proposals,
                tic_id=tic_id,
                sector=sector,
                quality_base={
                    "role": "repeat-search-proposal",
                    "source": "TESS-SPOC FFI",
                },
            )
            events = list(records.values())
            candidate_events.extend(events)
            sector_results.append(
                {
                    "sector": sector,
                    "status": "complete",
                    "proposals": len(proposals),
                    "events": [event_summary(event) for event in events],
                    "skipped": [entry.reason for entry in skipped],
                }
            )
        except Exception as error:  # noqa: BLE001 — preserve per-sector results
            sector_results.append(
                {"sector": sector, "status": "failed", "reason": str(error)[:300]}
            )

    ranked = rank_repeat_events(anchor_result, candidate_events, THRESHOLDS)
    return {
        "tic_id": tic_id,
        "available_sectors": sectors,
        "coverage_error": coverage_error,
        "status": "complete",
        "anchor": event_summary(anchor_result),
        "sectors": sector_results,
        "ranked_repeats": ranked,
        "compatible_repeats": [entry for entry in ranked if entry["compatible"]],
        "fetched": fetched,
    }


def _render(results: dict) -> str:
    lines = [
        "# Candidate repeat-event search",
        f"Run: {results['run_utc']}",
        "",
        "The search reprocessed every available TESS-SPOC FFI sector for each "
        "isolated-event candidate. All proposals remain in the JSON; the ranked "
        "queue applies the frozen deterministic matcher and reports timing failures.",
        "",
    ]
    for result in results["candidates"]:
        lines.extend(
            [
                f"## TIC {result['tic_id']}",
                f"Status: {result['status']}; sectors: {result['available_sectors']}.",
            ]
        )
        if result["status"] != "complete":
            lines.append(f"Reason: {result.get('reason', 'archive unavailable')}.")
            continue
        lines.append(
            f"Anchor: S{result['anchor']['sector']} @ {result['anchor']['t0']:.5f}; "
            f"{len(result['ranked_repeats'])} cross-sector event(s) ranked, "
            f"{len(result['compatible_repeats'])} compatible."
        )
        for entry in result["ranked_repeats"][:10]:
            event = entry["event"]
            lines.append(
                f"- S{event['sector']} @ {event['t0']:.5f}: "
                f"{'compatible' if entry['compatible'] else 'rejected'}; "
                f"score {entry['score']}; {entry['explanation']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "product": "TESS-SPOC FFI",
        "thresholds": THRESHOLDS,
        "candidates": [_search_case(case) for case in CASES],
    }
    output = Path("reports")
    output.mkdir(exist_ok=True)
    (output / "candidate_repeat_search.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (output / "candidate_repeat_search.md").write_text(_render(results))
    print(_render(results))
    print("wrote reports/candidate_repeat_search.json and .md")


if __name__ == "__main__":
    main()
