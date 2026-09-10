"""Tracer-bullet end-to-end pipeline (issue #2).

Manifest → event records → candidate pairs → deterministic matches →
period aliases → window filtering → machine-readable results + report.
Pure functions on the manifest dict; no archive access, no ML.
"""

from __future__ import annotations

from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc.event import EventRecord
from tess_assoc.freeze_context import FrozenRunContext
from tess_assoc.manifest import TracerManifest, load_manifest
from tess_assoc.matcher import match
from tess_assoc.pairs import build_pairs
from tess_assoc.provider import provide_events
from tess_assoc.window import filter_aliases, samples_in_windows


def _stage_results(
    manifest: TracerManifest,
    events: dict[str, EventRecord],
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
                    "aliases_retained": sum(v.retained for v in verdicts),
                    "aliases_rejected": sum(not v.retained for v in verdicts),
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

    touched = (
        {s.sector for s in manifest.sectors}
        | {e.sector for e in manifest.events}
    )
    return pair_results, associations, records, touched


def _validate_manifest(manifest: TracerManifest) -> TracerManifest:
    if not isinstance(manifest, TracerManifest):
        raise ValueError("manifest must be a TracerManifest")
    return manifest


def _validate_event_inputs(
    manifest: TracerManifest, events: dict[str, EventRecord]
) -> list[EventRecord]:
    if not isinstance(events, dict):
        raise ValueError("events must be a dict")
    if any(not isinstance(event_id, str) or not event_id for event_id in events):
        raise ValueError("event ids must be non-empty strings")
    records = list(events.values())
    if not all(isinstance(record, EventRecord) for record in records):
        raise ValueError("events must map ids to EventRecords")
    record_tics = {record.tic_id for record in records}
    if record_tics and record_tics != {manifest.tic_id}:
        raise ValueError("event records must belong to the manifest TIC")
    manifest_sectors = {sector.sector for sector in manifest.sectors}
    undeclared_sectors = {record.sector for record in records} - manifest_sectors
    if undeclared_sectors:
        raise ValueError(
            "event records contain sectors not declared by the manifest: "
            f"{sorted(undeclared_sectors)}"
        )
    windows_by_sector = {
        sector.sector: sector.windows for sector in manifest.sectors
    }
    for record in records:
        if not any(
            start <= record.t0 <= end
            for start, end in windows_by_sector[record.sector]
        ):
            raise ValueError(
                f"event record t0 outside declared sector {record.sector} windows"
            )
        if not samples_in_windows(record.local_time, windows_by_sector[record.sector]):
            raise ValueError(
                f"event record samples outside declared sector {record.sector} windows"
            )
    return records


def _validate_development_records(records: list[EventRecord]) -> None:
    _protocol.validate_development_sectors({record.sector for record in records})


def _run_validated_records(
    manifest: TracerManifest,
    events: dict[str, EventRecord],
) -> dict[str, Any]:
    pair_results, associations, records, touched = _stage_results(manifest, events)
    return {
        "fixture": manifest.name,
        "tic_id": manifest.tic_id,
        "protocol_version": _protocol.PROTOCOL_VERSION,
        "events": [e.to_dict() for e in records],
        "pairs": pair_results,
        "associations": associations,
        "sealed_sectors_touched": sorted(touched & set(_protocol.SEALED_SECTORS)),
    }


def run_records(
    manifest: TracerManifest, events: dict[str, EventRecord]
) -> dict[str, Any]:
    """Core stages over prebuilt records (shared by fixture and replay paths)."""
    manifest = _validate_manifest(manifest)
    manifest.validate_development()
    _validate_development_records(_validate_event_inputs(manifest, events))
    return _run_validated_records(manifest, events)


def run_frozen_records(
    manifest: TracerManifest,
    events: dict[str, EventRecord],
    *,
    context: FrozenRunContext,
    system_payload: dict[str, Any],
) -> dict[str, Any]:
    """Same core stages over gated data — verified freeze required.

    Rejects sealed sectors exactly like run_records unless the freeze
    record verifies (same source tree, same thresholds). The freeze
    evidence lands in the output for audit.
    """
    if not isinstance(context, FrozenRunContext):
        raise ValueError("run_frozen_records requires a FrozenRunContext")
    context.require_unblinded()
    manifest = _validate_manifest(manifest)
    _validate_event_inputs(manifest, events)
    sectors = {s.sector for s in manifest.sectors} | {
        e.sector for e in manifest.events
    }
    context.check_system(manifest.tic_id, sectors, payload=system_payload)
    context.check_thresholds(dict(manifest.matcher_thresholds), "frozen")
    pair_results, associations, records, touched = _stage_results(manifest, events)
    return {
        "fixture": manifest.name,
        "tic_id": manifest.tic_id,
        "protocol_version": _protocol.PROTOCOL_VERSION,
        "events": [e.to_dict() for e in records],
        "pairs": pair_results,
        "associations": associations,
        "sealed_sectors_touched": sorted(touched & set(_protocol.SEALED_SECTORS)),
        "freeze": context.evidence(),
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
                     f"{asc['aliases_total']} aliases "
                     f"({asc['aliases_retained']} retained, "
                     f"{asc['aliases_rejected']} rejected)")
        kept = ", ".join(f"n={r['n']} P={r['period_days']:.1f}d" for r in asc["retained"])
        cut = ", ".join(
            f"n={r['n']} P={r['period_days']:.1f}d (missing epoch "
            f"{r['contradicting_epoch']:.1f})" for r in asc["rejected"]
        )
        lines.append(f"  retained: {kept}")
        lines.append(f"  rejected: {cut}")
    lines.append("")
    lines.append("## Event evidence")
    for index, event in enumerate(results.get("events", [])):
        evidence = event.get("observability")
        lines.append(f"- event {index}: {evidence}")
    skipped = results.get("skipped", [])
    if skipped:
        lines.append("")
        lines.append("## Skipped proposals")
        for entry in skipped:
            lines.append(
                f"- sector {entry['sector']} t0={entry['t0']}: "
                f"{entry['reason']} — {entry.get('observability')}"
            )
    missed = results.get("missed", [])
    if missed:
        lines.append("")
        lines.append("## Missed transits")
        for entry in missed:
            lines.append(
                f"- sector {entry['sector']} t0={entry['t0']}: "
                f"{entry['reason']} — {entry.get('observability')}"
            )
    lines.append("")
    lines.append(f"Sealed sectors touched: {results['sealed_sectors_touched']}")
    return "\n".join(lines) + "\n"


def run_tracer(manifest: TracerManifest) -> dict[str, Any]:
    manifest = _validate_manifest(manifest)
    manifest.validate_development()
    events = provide_events(manifest)
    _validate_development_records(_validate_event_inputs(manifest, events))
    return _run_validated_records(manifest, events)


def run_tracer_dict(manifest_dict: dict[str, Any]) -> dict[str, Any]:
    return run_tracer(load_manifest(manifest_dict))
