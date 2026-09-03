"""Tracer-bullet end-to-end pipeline (issue #2).

Manifest → event records → candidate pairs → deterministic matches →
period aliases → window filtering → machine-readable results + report.
Pure functions on the manifest dict; no archive access, no ML.
"""

from __future__ import annotations

from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc.event import EventRecord
from tess_assoc import freeze as _freeze
from tess_assoc.manifest import TracerManifest, load_manifest
from tess_assoc.matcher import match
from tess_assoc.pairs import build_pairs
from tess_assoc.provider import provide_events
from tess_assoc.window import filter_aliases


def _stage_results(
    manifest: TracerManifest, events: dict[str, EventRecord]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[EventRecord], set[int]]:
    """Shared core: pairs → deterministic matches → alias filtering."""
    records = list(events.values())
    pairs = build_pairs(events)
    thresholds = manifest.matcher_thresholds

    pair_results: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for p in pairs:
        a, b = events[p.a_id], events[p.b_id]
        decision = match(a, b, thresholds)
        entry: dict[str, Any] = {
            "a": p.a_id,
            "b": p.b_id,
            "compatible": decision.compatible,
            "rel_depth_diff": decision.rel_depth_diff,
            "rel_duration_diff": decision.rel_duration_diff,
            "morph_corr": decision.morph_corr,
            "timing_plausible": decision.timing_plausible,
            "explanation": decision.explanation,
        }
        pair_results.append(entry)
        if decision.compatible:
            t1, t2 = sorted([a.t0, b.t0])
            verdicts = filter_aliases(a, b, manifest, records)
            associations.append(
                {
                    "pair": [p.a_id, p.b_id],
                    "delta_t_days": t2 - t1,
                    "aliases_total": len(verdicts),
                    "retained": [
                        {"n": v.n, "period_days": v.period_days}
                        for v in verdicts
                        if v.retained
                    ],
                    "rejected": [
                        {
                            "n": v.n,
                            "period_days": v.period_days,
                            "contradicting_epoch": v.contradicting_epoch,
                        }
                        for v in verdicts
                        if not v.retained
                    ],
                }
            )

    touched = {s.sector for s in manifest.sectors} | {e.sector for e in manifest.events}
    return pair_results, associations, records, touched


def run_records(
    manifest: TracerManifest, events: dict[str, EventRecord]
) -> dict[str, Any]:
    """Core stages over prebuilt records (shared by fixture and replay paths)."""
    pair_results, associations, records, touched = _stage_results(manifest, events)
    _protocol.validate_no_temporal_leak(touched)
    return {
        "fixture": manifest.name,
        "tic_id": manifest.tic_id,
        "protocol_version": _protocol.PROTOCOL_VERSION,
        "events": [e.to_dict() for e in records],
        "pairs": pair_results,
        "associations": associations,
        "sealed_sectors_touched": sorted(touched & set(_protocol.SEALED_SECTORS)),
    }


def run_holdout_records(
    manifest: TracerManifest,
    events: dict[str, EventRecord],
    *,
    freeze_record,
) -> dict[str, Any]:
    """Same core stages over sealed data — verified freeze required.

    Rejects sealed sectors exactly like run_records unless the freeze
    record verifies (same source tree, same thresholds). The freeze
    evidence lands in the output for audit.
    """
    if dict(manifest.matcher_thresholds) != freeze_record.thresholds:
        raise ValueError("holdout thresholds differ from frozen thresholds")
    if _freeze.source_tree_hash() != freeze_record.code_sha:
        raise ValueError("source tree changed since freeze")
    pair_results, associations, records, touched = _stage_results(manifest, events)
    return {
        "fixture": manifest.name,
        "tic_id": manifest.tic_id,
        "protocol_version": _protocol.PROTOCOL_VERSION,
        "events": [e.to_dict() for e in records],
        "pairs": pair_results,
        "associations": associations,
        "sealed_sectors_touched": sorted(touched & set(_protocol.SEALED_SECTORS)),
        "freeze": {
            "code_sha": freeze_record.code_sha,
            "created_utc": freeze_record.created_utc,
            "unblinded_utc": freeze_record.unblinded_utc,
        },
    }


def render_report(results: dict[str, Any]) -> str:
    lines = [
        f"# Tracer report: {results['fixture']} (TIC {results['tic_id']})",
        f"Protocol {results['protocol_version']}; "
        f"{len(results['events'])} events, {len(results['pairs'])} pairs, "
        f"{len(results['associations'])} associations.",
        "",
        "## Pairs",
    ]
    for p in results["pairs"]:
        flag = "COMPATIBLE" if p["compatible"] else "incompatible"
        lines.append(f"- {p['a']}–{p['b']}: {flag} — {p['explanation']}")
    lines.append("")
    lines.append("## Associations")
    for asc in results["associations"]:
        a, b = asc["pair"]
        lines.append(f"- {a}–{b}: ΔT={asc['delta_t_days']:.1f}d, "
                     f"{asc['aliases_total']} aliases")
        kept = ", ".join(f"n={r['n']} P={r['period_days']:.1f}d" for r in asc["retained"])
        cut = ", ".join(
            f"n={r['n']} P={r['period_days']:.1f}d (missing epoch "
            f"{r['contradicting_epoch']:.1f})" for r in asc["rejected"]
        )
        lines.append(f"  retained: {kept}")
        lines.append(f"  rejected: {cut}")
    lines.append("")
    lines.append(f"Sealed sectors touched: {results['sealed_sectors_touched']}")
    return "\n".join(lines) + "\n"


def run_tracer(manifest: TracerManifest) -> dict[str, Any]:
    return run_records(manifest, provide_events(manifest))


def run_tracer_dict(manifest_dict: dict[str, Any]) -> dict[str, Any]:
    return run_tracer(load_manifest(manifest_dict))
