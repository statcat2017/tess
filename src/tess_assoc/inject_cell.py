"""One injection cell: inject a pair, propose blind, match, classify.

A cell injects two transit dips (one per sector, or both into one light
curve for same-sector timing negatives), runs both modified curves through
the real blind proposer and shared extraction core, builds the full
candidate-pair set with deterministic ranking, and classifies the outcome.
"""

from __future__ import annotations

from typing import Any

from tess_assoc._validate import require_finite, require_positive_finite
from tess_assoc.benchmark import rank_pairs
from tess_assoc.event import EventRecord
from tess_assoc.extract import coverage_windows
from tess_assoc.inject_geometry import SHAPES
from tess_assoc.manifest import ManifestSector, TracerManifest
from tess_assoc.matcher import REQUIRED_THRESHOLDS, match, match_score
from tess_assoc.orbit import generate_aliases
from tess_assoc.pairs import build_pairs
from tess_assoc.propose import (
    PROPOSER_SNR_THRESHOLD,
    propose_events,
    records_from_proposals,
)
from tess_assoc.protocol import (
    LONG_PERIOD_LOWER_BOUND_DAYS,
    validate_no_temporal_leak,
)
from tess_assoc.provider import shape_flux
from tess_assoc.replay import RECALL_TOL_DAYS
from tess_assoc.window import filter_aliases


def inject_transit(
    time: list[float],
    flux: list[float],
    t0: float,
    depth: float,
    duration_days: float,
    shape: str = "box",
) -> list[float]:
    """Multiply a transit dip into a light curve (pure, deterministic)."""
    require_finite("t0", t0)
    require_positive_finite("depth", depth)
    require_positive_finite("duration_days", duration_days)
    if shape not in ("box", "v"):
        raise ValueError("shape must be 'box' or 'v'")
    if len(time) != len(flux):
        raise ValueError("time and flux must be equal length")
    return [
        f * shape_flux(shape, t - t0, depth, duration_days)
        for t, f in zip(time, flux)
    ]


def failure_mode(detected: bool, compatible: bool, expected_positive: bool) -> str:
    """Outcome class against the cell's own expectation (not just timing)."""
    if not detected:
        return "detection-failure"
    if compatible != expected_positive:
        return "association-failure"
    return "recovered"


def candidate_learn_payload(
    rec_a: EventRecord,
    rec_b: EventRecord,
    tic_id: int,
    label: int,
    candidate_ids: list[str],
) -> dict[str, Any]:
    """Single learned-scoring payload for one candidate pair (built once)."""
    return {
        "tic_id": tic_id,
        "fa": list(rec_a.local_flux),
        "fb": list(rec_b.local_flux),
        "feat_a": {
            "depth": rec_a.depth,
            "duration_days": rec_a.duration_days,
            "snr": rec_a.snr,
        },
        "feat_b": {
            "depth": rec_b.depth,
            "duration_days": rec_b.duration_days,
            "snr": rec_b.snr,
        },
        "label": label,
        "candidate": list(candidate_ids),
    }


def cell_group_key(
    tic_id: int,
    sector_a: int,
    sector_b: int,
    t1: float,
    delta_t_days: float,
    depth: float,
    shape: str,
    shape_b: str,
) -> str:
    """Unique identity for one injection experiment (separation included)."""
    return (
        f"{tic_id}:{sector_a}:{sector_b}:{t1:.12g}:{delta_t_days:.12g}:"
        f"{depth:.12g}:{shape}:{shape_b}"
    )


def best_distinct_target(
    recs_a: dict[str, EventRecord],
    recs_b: dict[str, EventRecord],
    near_a: list[str],
    near_b: list[str],
    t1: float,
    t2: float,
) -> tuple[str | None, str | None]:
    """Nearest distinct record pair covering both injection sites.

    The two sites are far apart in practice, so the nearest hits are
    trivially distinct; the explicit distinctness guard only matters for
    degenerate caller-supplied separations, where reusing one record for
    both sites would fake a self-match.
    """
    best: tuple[float, str, str] | None = None
    for rid_a in near_a:
        for rid_b in near_b:
            if rid_a == rid_b and recs_a is recs_b:
                continue
            cost = abs(recs_a[rid_a].t0 - t1) + abs(recs_b[rid_b].t0 - t2)
            if best is None or cost < best[0]:
                best = (cost, rid_a, rid_b)
    if best is None:
        return None, None
    return best[1], best[2]


def _require_source_provenance(provenance: dict[str, Any] | None) -> dict[str, Any]:
    if (
        not isinstance(provenance, dict)
        or "source_a" not in provenance
        or "source_b" not in provenance
        or not isinstance(provenance["source_a"], dict)
        or not isinstance(provenance["source_b"], dict)
    ):
        raise ValueError(
            "provenance must be a dict with source_a/source_b dicts "
            "(empty dicts allowed for synthetic curves; real runs carry "
            "data_uri, local_path, retrieved_utc)"
        )
    return dict(provenance)


def study_cell(
    time_a: list[float],
    flux_a: list[float],
    time_b: list[float],
    flux_b: list[float],
    *,
    tic_id: int,
    sector_a: int,
    sector_b: int,
    t1: float,
    delta_t_days: float,
    depth: float,
    duration_days: float,
    shape: str,
    thresholds: dict[str, float],
    shape_b: str | None = None,
    half_span_days: float = 0.6,
    resample_samples: int = 61,
    provenance: dict[str, Any] | None = None,
    alias_tol_days: float = 0.3,
    all_sector_times: dict[int, list[float]] | None = None,
    other_records: list[EventRecord] | None = None,
) -> dict[str, Any]:
    """One grid cell: inject a pair, propose blind, match, classify.

    Both modified curves go through the real blind proposer — detection is
    measured, never assumed. Same-sector cells inject both dips into one
    shared light curve so the proposer sees realistic event competition.
    """
    require_positive_finite("delta_t_days", delta_t_days)
    validate_no_temporal_leak((sector_a, sector_b))
    for key in REQUIRED_THRESHOLDS:
        if key not in thresholds:
            raise ValueError(f"thresholds missing key: {key}")
    base_provenance = _require_source_provenance(provenance)
    if all_sector_times is not None:
        validate_no_temporal_leak(all_sector_times)
    validate_no_temporal_leak(
        [record.sector for record in (other_records or [])]
    )
    shape_b = shape if shape_b is None else shape_b
    if shape not in SHAPES or shape_b not in SHAPES:
        raise ValueError("shape must be 'box' or 'v'")
    t2 = t1 + delta_t_days
    cross_epoch = sector_a != sector_b and delta_t_days >= LONG_PERIOD_LOWER_BOUND_DAYS
    expected_positive = cross_epoch and shape == shape_b
    if sector_a == sector_b:
        modified = inject_transit(time_a, flux_a, t1, depth, duration_days, shape)
        modified = inject_transit(time_a, modified, t2, depth, duration_days, shape_b)
        mod_a = mod_b = modified
    else:
        mod_a = inject_transit(time_a, flux_a, t1, depth, duration_days, shape)
        mod_b = inject_transit(time_b, flux_b, t2, depth, duration_days, shape_b)
    base = {
        **base_provenance,
        "role": "injected",
        "origin": "injection",
        "matcher_thresholds": dict(thresholds),
        "protocol_version": "v1",
        "sector_ids": [sector_a, sector_b],
        "proposer_snr_threshold": PROPOSER_SNR_THRESHOLD,
        "window_half_span_days": half_span_days,
        "resample_samples": resample_samples,
        "alias_tol_days": alias_tol_days,
        "injected_depth": depth,
        "injected_duration_days": duration_days,
        "injected_shape": shape,
        "injected_shape_b": shape_b,
        "injected_delta_t_days": delta_t_days,
        "injected_t0s": [t1, t2],
    }
    recs_a, _ = records_from_proposals(
        time_a, mod_a, propose_events(time_a, mod_a),
        tic_id=tic_id, sector=sector_a,
        half_span_days=half_span_days, resample_samples=resample_samples,
        quality_base={**base, "injected_t0": t1},
    )
    if sector_a == sector_b:
        recs_b = recs_a
    else:
        recs_b, _ = records_from_proposals(
            time_b, mod_b, propose_events(time_b, mod_b),
            tic_id=tic_id, sector=sector_b,
            half_span_days=half_span_days, resample_samples=resample_samples,
            quality_base={**base, "injected_t0": t2},
        )
    near_a = [rid for rid, r in recs_a.items() if abs(r.t0 - t1) <= RECALL_TOL_DAYS]
    near_b = [rid for rid, r in recs_b.items() if abs(r.t0 - t2) <= RECALL_TOL_DAYS]
    target_a_id, target_b_id = best_distinct_target(
        recs_a, recs_b, near_a, near_b, t1, t2
    )
    detected = target_a_id is not None and target_b_id is not None
    candidate_events = dict(recs_a)
    candidate_events.update(recs_b)
    candidate_entries: list[dict[str, Any]] = []
    learn_candidates: list[dict[str, Any]] = []
    for candidate in build_pairs(candidate_events):
        rec_pair_a = candidate_events[candidate.a_id]
        rec_pair_b = candidate_events[candidate.b_id]
        candidate_decision = match(rec_pair_a, rec_pair_b, thresholds)
        candidate_entries.append(
            {
                "a": candidate.a_id,
                "b": candidate.b_id,
                "score": match_score(candidate_decision),
                "label": "negative",
            }
        )
        learn_candidates.append(
            candidate_learn_payload(
                rec_pair_a, rec_pair_b, tic_id, 0,
                [candidate.a_id, candidate.b_id],
            )
        )
    target_entry = None
    if detected:
        for entry in candidate_entries:
            if {entry["a"], entry["b"]} == {target_a_id, target_b_id}:
                entry["label"] = "positive" if expected_positive else "negative"
                target_entry = entry
                break
        for learned_candidate in learn_candidates:
            if set(learned_candidate["candidate"]) == {target_a_id, target_b_id}:
                learned_candidate["label"] = 1 if expected_positive else 0
    candidate_ranking = rank_pairs(candidate_entries)
    group = cell_group_key(
        tic_id, sector_a, sector_b, t1, delta_t_days, depth, shape, shape_b
    )
    for learned_candidate in learn_candidates:
        learned_candidate["group"] = group
        learned_candidate["target"] = (
            target_entry is not None
            and set(learned_candidate["candidate"]) == {target_a_id, target_b_id}
        )
    decision = None
    learn = None
    aliases_before, aliases_after = 0, 0
    alias_status = "not-evaluated"
    if detected:
        ra, rb = recs_a[target_a_id], recs_b[target_b_id]
        decision = match(ra, rb, thresholds)
        matched = next(
            c for c in learn_candidates if set(c["candidate"]) == {target_a_id, target_b_id}
        )
        learn = {
            "tic_id": tic_id,
            "fa": matched["fa"],
            "fb": matched["fb"],
            "feat_a": matched["feat_a"],
            "feat_b": matched["feat_b"],
            "det_score": match_score(decision),
            "label": 1 if expected_positive else 0,
        }
        if expected_positive and decision.compatible:
            alias_status = "evaluated"
            sector_times = all_sector_times or {
                sector_a: time_a,
                sector_b: time_b,
            }
            manifest = TracerManifest(
                name="injection-alias",
                tic_id=tic_id,
                epoch_match_tol_days=alias_tol_days,
                matcher_thresholds=dict(thresholds),
                sectors=tuple(
                    ManifestSector(sector=s, windows=tuple(coverage_windows(t)))
                    for s, t in sorted(sector_times.items())
                ),
                events=tuple(),
            )
            observed_records = list(other_records or [])
            observed_records.extend(recs_a.values())
            if sector_b != sector_a:
                observed_records.extend(recs_b.values())
            verdicts = filter_aliases(ra, rb, manifest, observed_records)
            aliases_before = len(generate_aliases(abs(rb.t0 - ra.t0)))
            aliases_after = sum(1 for v in verdicts if v.retained)
    failure = failure_mode(
        detected, decision.compatible if decision else False, expected_positive
    )
    return {
        "tic_id": tic_id,
        "sectors": [sector_a, sector_b],
        "t1": t1,
        "t2": t2,
        "delta_t_days": delta_t_days,
        "separation": "cross-epoch" if cross_epoch else "same-epoch",
        "depth": depth,
        "duration_days": duration_days,
        "shape": shape,
        "shape_b": shape_b,
        "pair_type": (
            "positive-repeat" if expected_positive
            else "morphology-negative" if cross_epoch
            else "timing-negative"
        ),
        "expected_positive": expected_positive,
        "n_proposals_a": len(recs_a),
        "n_proposals_b": len(recs_b),
        "n_candidate_pairs": candidate_ranking["ranked_n"],
        "target_rank": target_entry.get("rank") if target_entry else None,
        "learn_candidates": learn_candidates,
        "detected": detected,
        "compatible": decision.compatible if decision else False,
        "failure_mode": failure,
        "score": match_score(decision) if decision else float("-inf"),
        "morph_corr": decision.morph_corr if decision else 0.0,
        "aliases_before": aliases_before,
        "aliases_after": aliases_after,
        "alias_status": alias_status,
        "provenance": base,
        "learn": learn,
    }


__all__ = [
    "best_distinct_target",
    "candidate_learn_payload",
    "cell_group_key",
    "failure_mode",
    "inject_transit",
    "study_cell",
]
