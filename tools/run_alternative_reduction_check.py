"""Compare QLP, TARS, and TGLC reductions for TIC 117549174."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.alternative import (
    DEFAULT_PROVIDERS,
    download_product,
    event_record_from_curve,
    measure_event,
    query_alternative_products,
    read_lightcurve,
)
from tess_assoc.candidate_followup import event_summary, rank_repeat_events
from tess_assoc.matcher import REQUIRED_THRESHOLDS
from tess_assoc.observability import SourceProduct
from tess_assoc.propose import propose_with_detail, records_from_proposals


TIC_ID = 117549174
ANCHOR_SECTOR = 69
ANCHOR_T0 = 3183.133477153529
ANCHOR_DURATION_DAYS = 0.06944716061070721
SECTORS = (2, 29, 69, 70)
CACHE = Path("/tmp/alternative_cache")
THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}


def _provider_check(provider: str, products: list) -> dict:
    selected = [product for product in products if product.provider == provider]
    downloaded = {}
    failures = []
    for product in selected:
        try:
            path = download_product(product, CACHE)
            downloaded[product.sector] = (product, path)
        except Exception as error:  # noqa: BLE001 — preserve provider coverage
            failures.append({"sector": product.sector, "reason": str(error)[:300]})
    anchor = downloaded.get(ANCHOR_SECTOR)
    if anchor is None:
        return {
            "provider": provider,
            "status": "no-anchor-sector",
            "products": [product.__dict__ for product in selected],
            "download_failures": failures,
        }

    anchor_product, anchor_path = anchor
    anchor_curve = read_lightcurve(
        anchor_path,
        source_product=SourceProduct(
            anchor_product.provider,
            "alternative light curve",
            anchor_product.data_uri,
            "not-recorded",
        ),
    )
    anchor_measurement = measure_event(
        anchor_curve, t0=ANCHOR_T0, duration_days=ANCHOR_DURATION_DAYS
    )
    anchor_record = event_record_from_curve(
        anchor_curve,
        tic_id=TIC_ID,
        sector=ANCHOR_SECTOR,
        t0=ANCHOR_T0,
        duration_days=ANCHOR_DURATION_DAYS,
        role=f"{provider.lower()}-anchor",
    )
    sector_results = []
    candidates = []
    for sector, (product, path) in sorted(downloaded.items()):
        curve = read_lightcurve(
            path,
            source_product=SourceProduct(
                product.provider,
                "alternative light curve",
                product.data_uri,
                "not-recorded",
            ),
        )
        result = {
            "sector": sector,
            "path": str(path),
            "data_uri": product.data_uri,
            "flux_column": curve["flux_column"],
        }
        if sector == ANCHOR_SECTOR:
            result["anchor_measurement"] = anchor_measurement
            sector_results.append(result)
            continue
        try:
            proposals, _, _ = propose_with_detail(curve["time"], curve["flux"])
            records, skipped = records_from_proposals(
                curve["time"],
                curve["flux"],
                proposals,
                tic_id=TIC_ID,
                sector=sector,
                quality_base={"role": f"{provider.lower()}-repeat-search"},
                observability=curve["observability"],
            )
            candidates.extend(records.values())
            result["proposals"] = len(proposals)
            result["events"] = [event_summary(event) for event in records.values()]
            result["skipped"] = [entry.reason for entry in skipped]
        except Exception as error:  # noqa: BLE001 — preserve per-sector result
            result["status"] = "failed"
            result["reason"] = str(error)[:300]
        sector_results.append(result)

    ranked = (
        rank_repeat_events(anchor_record, candidates, THRESHOLDS)
        if anchor_record is not None
        else []
    )
    return {
        "provider": provider,
        "status": "complete" if anchor_record is not None else "anchor-unmeasurable",
        "products": [product.__dict__ for product in selected],
        "download_failures": failures,
        "anchor_measurement": anchor_measurement,
        "sectors": sector_results,
        "ranked_repeats": ranked,
        "compatible_repeats": [entry for entry in ranked if entry["compatible"]],
    }


def render(results: dict) -> str:
    lines = [
        "# TIC 117549174 alternative-reduction check",
        f"Run: {results['run_utc']}",
        "",
        "The Sector 69 event and all available non-anchor sectors were checked "
        "provider by provider. No provider result is a planet confirmation.",
        "",
    ]
    for result in results["providers"]:
        lines.append(f"## {result['provider']}: {result['status']}")
        if "anchor_measurement" in result:
            event = result["anchor_measurement"]["event"]
            lines.append(
                f"Anchor: {event['status']}; depth {event.get('depth')}; "
                f"SNR {event.get('snr')}; flux column {result['anchor_measurement']['flux_column']}."
            )
        else:
            lines.append("No measurable Sector 69 anchor product was available.")
        if "ranked_repeats" in result:
            lines.append(
                f"Repeat queue: {len(result['ranked_repeats'])} ranked, "
                f"{len(result['compatible_repeats'])} compatible."
            )
        for sector in result.get("sectors", []):
            if sector["sector"] != ANCHOR_SECTOR:
                lines.append(
                    f"- Sector {sector['sector']}: {sector.get('proposals', 0)} proposals."
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    products = query_alternative_products(TIC_ID, SECTORS, providers=DEFAULT_PROVIDERS)
    results = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "tic_id": TIC_ID,
        "anchor": {
            "sector": ANCHOR_SECTOR,
            "t0": ANCHOR_T0,
            "duration_days": ANCHOR_DURATION_DAYS,
        },
        "sectors": list(SECTORS),
        "providers_requested": list(DEFAULT_PROVIDERS),
        "thresholds": THRESHOLDS,
        "providers": [_provider_check(provider, products) for provider in DEFAULT_PROVIDERS],
    }
    output = Path("reports")
    output.mkdir(exist_ok=True)
    (output / "tic117549174_alternative_reductions.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (output / "tic117549174_alternative_reductions.md").write_text(render(results))
    print(render(results))
    print("wrote reports/tic117549174_alternative_reductions.json and .md")


if __name__ == "__main__":
    main()
