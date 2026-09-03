"""Sealed temporal-holdout evaluation (issue #8).

Runs the frozen pipeline on the sealed cohort: blind replay of each
system, candidate-pair benchmark, deterministic ranking, learned scoring
with the pre-unblinding frozen checkpoint, alias/window filtering, and a
stop/go-style verdict. No threshold, weight, or ranking rule can change
after unblinding — the freeze record is verified before the first sealed
byte is read, and the verification evidence is embedded in the results.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from tess_assoc import freeze as _freeze
from tess_assoc.benchmark import build_benchmark, rank_pairs
from tess_assoc.event import EventRecord
from tess_assoc.extract import predicted_transits
from tess_assoc.learn import (
    average_precision,
    decide,
    prepare_split,
    predict_proba,
    recall_at_far,
)
from tess_assoc.pipeline import run_holdout_records
from tess_assoc.replay import replay_blind_system


def build_holdout_systems(
    blind_results: dict[str, dict[str, Any]], manifest
) -> list[dict[str, Any]]:
    """Benchmark systems from blind replay output (same shape as dev path)."""
    by_tic = {s.tic_id: s for s in manifest.systems}
    systems = []
    for name, res in blind_results.items():
        tic = res["tic_id"]
        system = by_tic[tic]
        records = {
            f"k{i}": EventRecord.from_dict(e) for i, e in enumerate(res["events"])
        }
        spans = [(w[0], w[1]) for s in res["sectors"] for w in s["windows"]]
        known = [
            t
            for lo, hi in spans
            for t in predicted_transits(system.t0_bjd_tdb, system.period_days, lo, hi)
        ]
        systems.append(
            {
                "tic_id": tic,
                "records": records,
                "known": known,
                "sectors": res["sectors"],
            }
        )
    return systems


def holdout_metrics(
    entries: list[dict[str, Any]], scores: list[float], far: float
) -> dict[str, float]:
    """Ranking metrics on one scored entry list (both methods, same code)."""
    labels = [1 if e["label"] == "positive" else 0 for e in entries]
    rescored = [dict(e, score=s) for e, s in zip(entries, scores)]
    ranking = rank_pairs(rescored)
    ranks = ranking["positive_ranks"]
    return {
        "ap": average_precision(labels, scores),
        "mrr": ranking["mrr"],
        "burden_at_full_recall": float(ranking["burden_at_full_recall"]),
        "top1": sum(1 for r in ranks if r <= 1) / len(ranks) if ranks else 0.0,
        "top5": sum(1 for r in ranks if r <= 5) / len(ranks) if ranks else 0.0,
        "recall_at_far": recall_at_far(labels, scores, far)["recall"],
    }


def run_holdout(
    manifest,
    *,
    freeze_path: str,
    checkpoint: dict[str, Any],
    ablation: str,
    config,
    cache_dir: str | None = None,
    log_path: str | None = None,
) -> dict[str, Any]:
    """Frozen evaluation on the sealed cohort (gate first, metrics after)."""
    record = _freeze.verify_freeze(freeze_path, config)
    if record.ablation != ablation:
        raise ValueError("holdout ablation differs from frozen ablation")
    if dict(manifest.matcher_thresholds) != record.thresholds:
        raise ValueError("holdout thresholds differ from frozen thresholds")
    if record.checkpoint_sha is None:
        raise ValueError("freeze record holds no checkpoint hash")
    if _freeze.checkpoint_hash(checkpoint) != record.checkpoint_sha:
        raise ValueError("checkpoint weights differ from frozen checkpoint")
    record = _freeze.mark_unblinded(freeze_path)
    thresholds = dict(manifest.matcher_thresholds)

    runner = functools.partial(run_holdout_records, freeze_record=record)
    blind_results = {
        system.name: replay_blind_system(
            manifest, system, cache_dir, records_runner=runner
        )
        for system in manifest.systems
    }
    systems = build_holdout_systems(blind_results, manifest)
    partitions = {s["tic_id"]: "test" for s in systems}
    bench = build_benchmark(systems, partitions, thresholds)

    test_entries = [
        e for e in bench["positives"] + bench["negatives"]
        if e.get("tic_id") in partitions
    ]
    det_metrics = holdout_metrics(
        test_entries,
        [e["score"] for e in test_entries],
        config.false_association_rate,
    )
    split = prepare_split(
        bench["positives"] + bench["negatives"],
        systems,
        partitions,
        ablation,
        (),
        ("test",),
    )
    probas = predict_proba(checkpoint, split.test, ablation)
    learned_metrics = holdout_metrics(
        [
            {"label": ("positive" if r["label"] else "negative")}
            for r in split.test
        ],
        probas,
        config.false_association_rate,
    )
    n_test_positives = sum(r["label"] for r in split.test)
    verdict = decide(det_metrics, learned_metrics, config, n_test_positives)

    from tess_assoc import protocol as _protocol

    touched = {
        entry["sector"]
        for res in blind_results.values()
        for entry in res["sectors"]
    }
    sealed = sorted(touched & set(_protocol.SEALED_SECTORS))
    results = {
        "protocol_version": record.protocol_version,
        "freeze": {
            "code_sha": record.code_sha,
            "created_utc": record.created_utc,
            "unblinded_utc": record.unblinded_utc,
            "checkpoint_sha": record.checkpoint_sha,
            "ablation": ablation,
        },
        "cohort": {
            name: {
                "tic_id": res["tic_id"],
                "sectors": [s["sector"] for s in res["sectors"]],
                "n_proposals": res["n_proposals"],
                "recall": res["recall"],
                "pair_outcome": res["pair_outcome"],
            }
            for name, res in blind_results.items()
        },
        "sealed_sectors_touched": sealed,
        "n_test_positives": n_test_positives,
        "n_test_total": len(split.test),
        "deterministic": det_metrics,
        "learned": learned_metrics,
        "alias_reduction": bench["alias_reduction"],
        "verdict": verdict,
    }
    if log_path is not None:
        _freeze.log_access(
            log_path,
            {
                "event": "holdout_run",
                "freeze_code_sha": record.code_sha,
                "unblinded_utc": record.unblinded_utc,
                "systems": sorted(blind_results),
                "sealed_sectors_touched": sealed,
                "verdict": verdict["verdict"],
            },
        )
    return results


def render_holdout_report(results: dict[str, Any]) -> str:
    """Human-readable sealed evaluation (supports any of the 3 conclusions)."""
    det, learned = results["deterministic"], results["learned"]
    verdict = results["verdict"]
    lines = [
        f"# Temporal holdout report (protocol {results['protocol_version']})",
        f"Freeze {results['freeze']['code_sha'][:12]} "
        f"created {results['freeze']['created_utc']}; "
        f"unblinded {results['freeze']['unblinded_utc']}.",
        f"Sealed sectors touched: {results['sealed_sectors_touched']}",
        "",
        "## Cohort",
    ]
    for name, cohort in results["cohort"].items():
        recall = cohort["recall"]
        lines.append(
            f"- {name}: sectors {cohort['sectors']}, "
            f"{cohort['n_proposals']} proposals, "
            f"recall {recall['recalled']}/{recall['known']} "
            f"({recall['rate_coverable']:.2f} coverable), "
            f"anchor pair: {cohort['pair_outcome']}"
        )
    lines += [
        "",
        "## Head-to-head (identical holdout pairs)",
        f"Deterministic: AP {det['ap']:.3f}, MRR {det['mrr']:.3f}, "
        f"burden@{results['n_test_positives']}pos {det['burden_at_full_recall']:.0f}, "
        f"recall@FAR {det['recall_at_far']:.3f}.",
        f"Learned ({results['freeze']['ablation']}): AP {learned['ap']:.3f}, "
        f"MRR {learned['mrr']:.3f}, "
        f"burden@{results['n_test_positives']}pos "
        f"{learned['burden_at_full_recall']:.0f}, "
        f"recall@FAR {learned['recall_at_far']:.3f}.",
        "",
        "## Alias reduction (positive pairs)",
        f"median reduction: {results['alias_reduction']['median_reduction']:.3f}",
        "",
        f"## Verdict: {verdict['verdict']}",
    ]
    lines.extend(f"- {reason}" for reason in verdict["reasons"])
    lines.append(f"Scope: {verdict['scope']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "build_holdout_systems",
    "holdout_metrics",
    "render_holdout_report",
    "run_holdout",
]
