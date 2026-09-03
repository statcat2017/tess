"""Injection grid over cached sector curves (issue #7).

Runs the depth x shape x separation grid and summarizes detection
separately from conditional association. Same-epoch cells are
timing-impossible negatives; cross-epoch cells hold positive repeats plus
one morphology-mismatched negative per unordered shape pair.
"""

from __future__ import annotations

from typing import Any

from tess_assoc._validate import require_positive_finite
from tess_assoc.event import EventRecord
from tess_assoc.inject_cell import failure_mode as cell_failure_mode
from tess_assoc.inject_cell import study_cell
from tess_assoc.inject_geometry import (
    DEPTHS,
    SAME_EPOCH_DT_DAYS,
    SHAPES,
    INJECTION_DURATION_DAYS,
    supported_pair,
)
from tess_assoc.propose import propose_events, records_from_proposals
from tess_assoc.protocol import validate_no_temporal_leak


def _mismatch_shape_pairs(
    shapes: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Unordered distinct shape pairs, each generated exactly once."""
    return [
        (shapes[i], shapes[j])
        for i in range(len(shapes))
        for j in range(i + 1, len(shapes))
    ]


def _invoke_cell(
    curves: dict[int, dict[str, Any]],
    sector_times: dict[int, list[float]],
    source_records: dict[int, list[EventRecord]],
    sectors: list[int],
    *,
    tic_id: int,
    sector_a: int,
    sector_b: int,
    t1: float,
    delta_t_days: float,
    depth: float,
    duration_days: float,
    shape: str,
    shape_b: str,
    thresholds: dict[str, float],
    half_span_days: float,
) -> dict[str, Any]:
    """Single study_cell call with shared provenance/window plumbing."""
    excluded = (sector_a,) if sector_a == sector_b else (sector_a, sector_b)
    return study_cell(
        curves[sector_a]["time"], curves[sector_a]["flux"],
        curves[sector_b]["time"], curves[sector_b]["flux"],
        tic_id=tic_id,
        sector_a=sector_a, sector_b=sector_b,
        t1=t1, delta_t_days=delta_t_days,
        depth=depth, duration_days=duration_days,
        shape=shape, shape_b=shape_b,
        thresholds=thresholds, half_span_days=half_span_days,
        provenance={
            "source_a": curves[sector_a].get("provenance", {}),
            "source_b": curves[sector_b].get("provenance", {}),
        },
        all_sector_times=sector_times,
        other_records=[
            r for sector in sectors if sector not in excluded
            for r in source_records[sector]
        ],
    )


def run_grid(
    curves: dict[int, dict[str, Any]],
    *,
    tic_id: int,
    thresholds: dict[str, float],
    depths: tuple[float, ...] = DEPTHS,
    shapes: tuple[str, ...] = SHAPES,
    duration_days: float = INJECTION_DURATION_DAYS,
    half_span_days: float = 0.6,
) -> list[dict[str, Any]]:
    """Full depth x shape x separation grid over cached sector curves.

    `curves` maps sector -> {"time": [...], "flux": [...], "provenance": {...}}.
    Separation classes: same-epoch (12d, timing-impossible by design) and
    cross-epoch (span-mid difference, timing-plausible). Cross-epoch holds
    one same-shape repeat per shape plus one mismatch negative per
    unordered shape pair.
    """
    require_positive_finite("duration_days", duration_days)
    sectors = sorted(curves)
    validate_no_temporal_leak(sectors)
    for shape in shapes:
        if shape not in SHAPES:
            raise ValueError("shape must be 'box' or 'v'")
    sector_times = {sector: curves[sector]["time"] for sector in sectors}
    source_records: dict[int, list[EventRecord]] = {}
    for sector in sectors:
        curve = curves[sector]
        source_provenance = {
            **curve.get("provenance", {}),
            "role": "source-light-curve",
        }
        records, _ = records_from_proposals(
            curve["time"], curve["flux"],
            propose_events(curve["time"], curve["flux"]),
            tic_id=tic_id, sector=sector,
            half_span_days=half_span_days,
            quality_base=source_provenance,
        )
        source_records[sector] = list(records.values())
    mismatch_pairs = _mismatch_shape_pairs(shapes)
    rows: list[dict[str, Any]] = []
    for depth in depths:
        require_positive_finite("depth", depth)
        for shape in shapes:
            for sector_a in sectors:
                ta = curves[sector_a]["time"]
                t1 = supported_pair(ta, ta, SAME_EPOCH_DT_DAYS, half_span_days)
                if t1 is not None:
                    rows.append(
                        _invoke_cell(
                            curves, sector_times, source_records, sectors,
                            tic_id=tic_id, sector_a=sector_a, sector_b=sector_a,
                            t1=t1, delta_t_days=SAME_EPOCH_DT_DAYS,
                            depth=depth, duration_days=duration_days,
                            shape=shape, shape_b=shape,
                            thresholds=thresholds, half_span_days=half_span_days,
                        )
                    )
            for i, sector_a in enumerate(sectors):
                for sector_b in sectors[i + 1 :]:
                    ta = curves[sector_a]["time"]
                    tb = curves[sector_b]["time"]
                    delta = (tb[0] + tb[-1]) / 2.0 - (ta[0] + ta[-1]) / 2.0
                    t1 = supported_pair(ta, tb, delta, half_span_days)
                    if t1 is None:
                        continue
                    rows.append(
                        _invoke_cell(
                            curves, sector_times, source_records, sectors,
                            tic_id=tic_id, sector_a=sector_a, sector_b=sector_b,
                            t1=t1, delta_t_days=delta,
                            depth=depth, duration_days=duration_days,
                            shape=shape, shape_b=shape,
                            thresholds=thresholds, half_span_days=half_span_days,
                        )
                    )
                    for shape_a, shape_c in mismatch_pairs:
                        if shape_a != shape:
                            continue
                        rows.append(
                            _invoke_cell(
                                curves, sector_times, source_records, sectors,
                                tic_id=tic_id, sector_a=sector_a,
                                sector_b=sector_b,
                                t1=t1, delta_t_days=delta,
                                depth=depth, duration_days=duration_days,
                                shape=shape_a, shape_b=shape_c,
                                thresholds=thresholds,
                                half_span_days=half_span_days,
                            )
                        )
    return rows


def _row_expectation(row: dict[str, Any], sep: str) -> bool:
    return bool(row.get("expected_positive", sep == "cross-epoch"))


def summarize_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell detection and conditional association outcomes.

    Same-epoch cells are timing-impossible negatives. Cross-epoch cells
    include positive repeats and morphology-mismatched negatives.
    `association_recall` covers positives only (None when a cell holds no
    positives); `negative_rejection_rate` covers negatives separately so
    the two meanings never share one field.
    """
    cells: dict[tuple[float, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        sep = row.get(
            "separation",
            "same-epoch" if row["sectors"][0] == row["sectors"][1] else "cross-epoch",
        )
        cells.setdefault(
            (row["depth"], row["shape"], row.get("shape_b", row["shape"]), sep),
            [],
        ).append(row)
    summary = []
    for (depth, shape, shape_b, sep), members in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        expected = [_row_expectation(m, sep) for m in members]
        det_correct = sum(
            1
            for m, is_positive in zip(members, expected)
            if m["detected"] and m["compatible"] == is_positive
        )
        detected_positives = [
            m for m, is_positive in zip(members, expected)
            if is_positive and m["detected"]
        ]
        detected_negatives = [
            m for m, is_positive in zip(members, expected)
            if not is_positive and m["detected"]
        ]
        association_correct = sum(m["compatible"] for m in detected_positives)
        failures = [
            m.get("failure_mode")
            or cell_failure_mode(m["detected"], m["compatible"], is_positive)
            for m, is_positive in zip(members, expected)
        ]
        reductions = [
            (m["aliases_before"] - m["aliases_after"]) / m["aliases_before"]
            for m in members
            if m["detected"] and _row_expectation(m, sep) and m["aliases_before"]
        ]
        reductions.sort()
        summary.append(
            {
                "depth": depth,
                "shape": shape,
                "shape_b": shape_b,
                "separation": sep,
                "n": len(members),
                "detection_recall": (
                    sum(1 for m in members if m["detected"]) / len(members)
                ),
                "association_evaluable_n": len(detected_positives),
                "association_recall": (
                    association_correct / len(detected_positives)
                    if detected_positives else None
                ),
                "negative_rejection_rate": (
                    sum(not m["compatible"] for m in detected_negatives)
                    / len(detected_negatives)
                    if detected_negatives else None
                ),
                "deterministic_correct_rate": det_correct / len(members),
                "detection_failures": sum(f == "detection-failure" for f in failures),
                "association_failures": sum(
                    f == "association-failure" for f in failures
                ),
                "median_alias_reduction": reductions[len(reductions) // 2] if reductions else 0.0,
            }
        )
    return summary


__all__ = ["run_grid", "summarize_grid"]
