"""Run Survey A: mask known planets and search hosts for additional events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tess_assoc.bulk import bulk_fetch, fetch_sector_script, parse_sector_script
from tess_assoc.freeze import create_freeze, file_hash
from tess_assoc.hosts import query_known_planets, select_known_host_targets
from tess_assoc.learn import LearnConfig
from tess_assoc.protocol import DEV_SECTORS
from tess_assoc.survey import build_survey_manifest, render_survey_report, run_mining_survey


THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}


def run(args: argparse.Namespace) -> dict:
    if args.last_sector not in DEV_SECTORS:
        raise ValueError("host search may only use development sectors 1-79")
    work = Path(args.work_dir)
    scripts = work / "scripts"
    cache = Path(args.cache_dir) if args.cache_dir else work / "cache"
    out = work / "out"
    scripts.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    sector_tics = {}
    for sector in range(1, args.last_sector + 1):
        script = fetch_sector_script(sector, str(scripts))
        sector_tics[sector] = parse_sector_script(script)

    query_utc = datetime.now(timezone.utc).isoformat()
    known_planets = query_known_planets(
        include_toi_candidates=not args.confirmed_only
    )
    targets = select_known_host_targets(
        known_planets,
        sector_tics,
        min_sectors=args.min_sectors,
        max_targets=args.max_targets,
    )
    manifest = build_survey_manifest(
        args.name,
        targets,
        thresholds=THRESHOLDS,
        purpose="mining",
        ephemeris_source="NASA Exoplanet Archive pscomppars + TOI known transit hosts",
    )
    manifest_path = work / "manifest.json"
    freeze_path = work / "freeze.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config = LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)
    create_freeze(
        str(Path(__file__).resolve().parents[1] / "fixtures" / "replay_v1.json"),
        str(manifest_path),
        config,
        output_path=str(freeze_path),
        cohort_key="discovery",
    )

    jobs = [
        (system["tic_id"], sector)
        for system in targets
        for sector in system["sectors"]
    ]
    fetched = bulk_fetch(jobs, str(cache), max_workers=args.download_workers)
    results = run_mining_survey(
        str(manifest_path),
        freeze_path=str(freeze_path),
        config=config,
        cache_dir=str(cache),
        out_dir=str(out),
        shortlist_k=args.shortlist_k,
        max_workers=args.survey_workers,
    )
    summary = {
        "known_planets": len(known_planets),
        "catalogue_queried_utc": query_utc,
        "catalogue_sources": [
            "NASA Exoplanet Archive pscomppars",
            "NASA Exoplanet Archive TOI" if not args.confirmed_only else "none",
        ],
        "runner_sha256": file_hash(str(Path(__file__).resolve())),
        "hosts_available": len({p.tic_id for p in known_planets}),
        "targets": len(targets),
        "recordings": len(jobs),
        "downloaded": len(fetched["downloaded"]),
        "cached": len(fetched["cached"]),
        "missing": len(fetched["missing"]),
        "failed": len(fetched["failed"]),
        "status": results["status"],
        "pairs_ranked": results["n_pairs_ranked"],
        "pair_candidates": len(results["candidates"]),
        "pair_reviewed": len(results["reviewed"]),
        "single_candidates": len(results["single_transits"]),
        "single_reviewed": len(results["single_transit_reviewed"]),
        "blocked": len(results["blocked_systems"]),
        "failed_systems": len(results["failed_systems"]),
        "manifest": str(manifest_path),
        "freeze": str(freeze_path),
    }
    (work / "fetch.json").write_text(json.dumps(fetched, indent=2) + "\n")
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (work / "report.md").write_text(render_survey_report(results) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/known_host_survey")
    parser.add_argument("--cache-dir")
    parser.add_argument("--name", default="known-hosts-development")
    parser.add_argument("--max-targets", type=int, default=500)
    parser.add_argument("--min-sectors", type=int, default=2)
    parser.add_argument("--last-sector", type=int, default=79)
    parser.add_argument("--confirmed-only", action="store_true")
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--survey-workers", type=int, default=8)
    parser.add_argument("--shortlist-k", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
