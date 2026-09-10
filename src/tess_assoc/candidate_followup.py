"""Pure ranking helpers for follow-up of isolated event candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from tess_assoc.event import EventRecord
from tess_assoc.matcher import match, match_score

FOLLOWUP_QUEUE_VERSION = "followup-v1"
PIXEL_OFFSET_REVIEW_THRESHOLD = 1.0

def event_summary(event: EventRecord) -> dict[str, Any]:
    """Return the serializable evidence needed for a repeat-event report."""
    return {
        "tic_id": event.tic_id,
        "sector": event.sector,
        "t0": event.t0,
        "depth": event.depth,
        "duration_days": event.duration_days,
        "snr": event.snr,
        "quality": dict(event.quality),
        "observability": (
            None
            if event.observability is None
            else event.observability.to_dict()
        ),
    }


def rank_repeat_events(
    anchor: EventRecord,
    candidates: Iterable[EventRecord],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Rank candidate events against an anchor without hiding timing failures.

    The report includes every candidate from another sector. Candidates whose
    separation is below the frozen 27-day single-star floor remain visible as
    timing failures, because that is useful evidence for eclipsing-binary and
    short-period false-positive diagnosis.
    """
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, EventRecord):
            raise ValueError("repeat candidates must be EventRecord records")
        if candidate.tic_id != anchor.tic_id:
            raise ValueError("repeat candidates must share the anchor TIC")
        if candidate.sector == anchor.sector:
            continue
        decision = match(anchor, candidate, thresholds)
        raw_score = match_score(decision)
        ranked.append(
            {
                "event": event_summary(candidate),
                "compatible": decision.compatible,
                "timing_plausible": decision.timing_plausible,
                "score": raw_score if math.isfinite(raw_score) else None,
                "rel_depth_diff": decision.rel_depth_diff,
                "rel_duration_diff": decision.rel_duration_diff,
                "morph_corr": decision.morph_corr,
                "explanation": decision.explanation,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["score"] is not None,
            item["score"] if item["score"] is not None else float("-inf"),
            item["event"]["snr"],
        ),
        reverse=True,
    )


def triage_followup(
    repeat_result: dict[str, Any],
    pixel_report: dict[str, Any],
) -> dict[str, Any]:
    """Convert repeat and pixel evidence into an explicit follow-up decision."""
    tic_id = int(repeat_result["tic_id"])
    if int(pixel_report["tic_id"]) != tic_id:
        raise ValueError("repeat and pixel reports must share a TIC")
    pixel = pixel_report["pixel"]
    ranked_repeats = repeat_result.get("ranked_repeats", [])
    if repeat_result.get("status") != "complete":
        repeat_status = "repeat-search-incomplete"
    elif not ranked_repeats:
        repeat_status = "no-qualifying-repeat"
    else:
        repeat_status = "repeat-events-found"

    centroid = pixel.get("centroid", {})
    target_apertures = pixel.get("apertures", [])
    measured_target_apertures = [
        aperture
        for aperture in target_apertures
        if aperture.get("status") == "measured"
    ]
    target_depths = [float(aperture["depth"]) for aperture in measured_target_apertures]
    target_reproduces_event = any(depth > 0 for depth in target_depths)
    offset = centroid.get("offset_pixels")
    if (
        pixel.get("difference", {}).get("status") == "measured"
        and isinstance(offset, (int, float))
        and offset >= PIXEL_OFFSET_REVIEW_THRESHOLD
        and not target_reproduces_event
    ):
        decision = "source-localisation-failure"
        priority = 2
        actions = [
            "obtain-speckle-imaging",
            "do-not-schedule-periodic-transit-follow-up",
            "reconsider-only-after-source-localisation",
        ]
    elif not measured_target_apertures:
        decision = "pixel-inconclusive"
        priority = 1
        actions = [
            "reprocess-raw-pixels-or-alternative-reduction",
            "do-not-promote-as-planet",
            "obtain-imaging-if-signal-survives-reprocessing",
        ]
    else:
        decision = "target-localisation-survives-initial-checks"
        priority = 1
        actions = [
            "obtain-speckle-imaging",
            "obtain-reconnaissance-spectroscopy",
            "schedule-multicolour-photometry-only-after-period-constraint",
        ]

    return {
        "tic_id": tic_id,
        "priority": priority,
        "decision": decision,
        "repeat_status": repeat_status,
        "repeat_count": len(ranked_repeats),
        "compatible_repeat_count": len(repeat_result.get("compatible_repeats", [])),
        "pixel_status": pixel.get("difference", {}).get("status"),
        "target_reproduces_event": target_reproduces_event,
        "difference_centroid_offset_pixels": offset,
        "actions": actions,
        "planet_claim": "blocked",
    }


def build_followup_queue(
    repeat_results: Iterable[dict[str, Any]],
    pixel_reports: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic queue from the two machine-readable reports."""
    pixels = {int(report["tic_id"]): report for report in pixel_reports}
    entries = []
    for repeat_result in repeat_results:
        tic_id = int(repeat_result["tic_id"])
        if tic_id not in pixels:
            raise ValueError(f"missing pixel report for TIC {tic_id}")
        entries.append(triage_followup(repeat_result, pixels[tic_id]))
    entries.sort(key=lambda entry: (entry["priority"], entry["tic_id"]))
    return {
        "queue_version": FOLLOWUP_QUEUE_VERSION,
        "entries": entries,
        "claim_boundary": "No entry is a confirmed planet; source localisation comes first.",
    }


__all__ = [
    "FOLLOWUP_QUEUE_VERSION",
    "build_followup_queue",
    "event_summary",
    "rank_repeat_events",
    "triage_followup",
]
