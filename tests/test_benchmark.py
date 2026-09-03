"""Candidate-pair benchmark tests (issue #5)."""

import json
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc import protocol as P
from tess_assoc.benchmark import (
    assign_partitions,
    build_benchmark,
    render_benchmark_report,
)
from tess_assoc.event import EventRecord
from tess_assoc.matcher import match, match_score

REPLAY = Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json"


def _rec(tic, sector, t0, depth=0.01, duration=0.2, shape="box"):
    n = 21
    times = [t0 - 0.3 + i * 0.03 for i in range(n)]
    if shape == "box":
        fluxes = [1.0 - depth if abs(t - t0) <= duration / 2 else 1.0 for t in times]
    else:
        fluxes = [1.0 - depth * max(0.0, 1.0 - abs(t - t0) / (duration / 2)) for t in times]
    return EventRecord(
        tic_id=tic, sector=sector, t0=t0, local_time=times, local_flux=fluxes,
        depth=depth, duration_days=duration, snr=10.0,
        stellar_meta={}, quality={},
    )


def _systems():
    recs_a = {
        "a1": _rec(1, 12, 100.0),
        "a2": _rec(1, 39, 1000.0),
        "ax": _rec(1, 39, 1005.0, depth=0.05, shape="v"),
        "am": _rec(1, 39, 1010.0, depth=0.02),
    }
    recs_b = {"b1": _rec(2, 12, 200.0), "b2": _rec(2, 39, 1100.0)}
    sectors = [{"sector": 12, "windows": [[90.0, 120.0]]}, {"sector": 39, "windows": [[990.0, 1020.0]]}]
    known_a = [100.0, 1000.0]
    known_b = [200.0, 1100.0]
    return [
        {"tic_id": 1, "records": recs_a, "known": known_a, "sectors": sectors},
        {"tic_id": 2, "records": recs_b, "known": known_b, "sectors": sectors},
    ]


THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}


def test_assign_partitions_split_and_complete():
    parts = assign_partitions([3, 1, 2])
    assert set(parts) == {1, 2, 3}
    assert set(parts.values()) == {"train", "validation", "test"}
    assert parts == assign_partitions([1, 2, 3])
    with pytest.raises(ValueError, match="partition leak"):
        P.validate_tic_partition({1, 2}, {2, 3})


def test_match_score_orders_correctly():
    good = match(_rec(1, 12, 100.0), _rec(1, 39, 1000.0), THRESHOLDS)
    bad = match(_rec(1, 12, 100.0), _rec(1, 39, 1005.0, depth=0.05, shape="v"), THRESHOLDS)
    timed_out = match(_rec(1, 12, 100.0), _rec(1, 12, 100.5), THRESHOLDS)
    assert match_score(good) > match_score(bad)
    assert match_score(timed_out) == float("-inf")


def test_benchmark_labels_slices_and_ranking():
    systems = _systems()
    parts = assign_partitions([1, 2])
    assert set(parts.values()) == {"train", "validation"}
    res = build_benchmark(systems, parts, THRESHOLDS)
    assert res["protocol_version"] == "v1"
    assert res["n_positives"] == 2  # exactly (a1,a2) and (b1,b2); ax/am are negatives
    assert res["partitions"] == parts
    cats = {n["category"] for n in res["negatives"]}
    assert "random" in cats
    assert "timing-incompatible" in cats  # same-sector pairs, 0.5-5d apart
    morph = res["slices"]["morphology-matched"]
    assert set(res["slices"]) == {
        "random", "morphology-matched", "timing-incompatible", "depth-duration-mismatch",
    }
    assert morph["n"] >= 1  # am: same box shape, 2x depth → corr high, incompatible
    assert res["ranking"]["mrr"] > 0
    assert res["ranking"]["burden_at_full_recall"] >= res["n_positives"]
    assert res["alias_reduction"]["median_reduction"] >= 0.0
    json.dumps(res)
    report = render_benchmark_report(res)
    assert "MRR" in report and "Slices" in report


def test_benchmark_rejects_unpartitioned_tic():
    with pytest.raises(ValueError, match="no partition"):
        build_benchmark(_systems(), {1: "train"}, THRESHOLDS)


def test_benchmark_rejects_bad_partitions_and_systems():
    systems = _systems()
    with pytest.raises(ValueError, match="unknown partition"):
        build_benchmark(systems, {1: "train", 2: "banana"}, THRESHOLDS)
    bad = [dict(systems[0], records="nope"), systems[1]]
    with pytest.raises(ValueError, match="non-empty dict"):
        build_benchmark(bad, assign_partitions([1, 2]), THRESHOLDS)
    with pytest.raises(ValueError, match="duplicate TIC"):
        assign_partitions([1, 1, 2])
    with pytest.raises(ValueError, match="duplicate TIC"):
        build_benchmark([systems[0], systems[0]], assign_partitions([1]), THRESHOLDS)


@needs_archive
def test_live_benchmark_three_systems(tmp_path):
    from tess_assoc.extract import predicted_transits
    from tess_assoc.replay import load_replay_manifest, replay_blind_system

    replay = load_replay_manifest(str(REPLAY))
    systems = []
    for system in replay.systems:
        res = replay_blind_system(replay, system, cache_dir=str(tmp_path))
        recs = {f"k{i}": EventRecord.from_dict(e) for i, e in enumerate(res["events"])}
        spans = [(w[0], w[1]) for s in res["sectors"] for w in s["windows"]]
        known = [
            t
            for lo, hi in spans
            for t in predicted_transits(system.t0_bjd_tdb, system.period_days, lo, hi)
        ]
        systems.append(
            {
                "tic_id": system.tic_id,
                "records": recs,
                "known": known,
                "sectors": res["sectors"],
            }
        )
    parts = assign_partitions([s["tic_id"] for s in systems])
    assert set(parts.values()) == {"train", "validation", "test"}
    res = build_benchmark(systems, parts, dict(replay.matcher_thresholds))
    assert res["n_positives"] >= 3
    assert res["ranking"]["mrr"] > 0
    assert res["alias_reduction"]["median_reduction"] >= 0.0
    json.dumps(res)
    assert "MRR" in render_benchmark_report(res)
