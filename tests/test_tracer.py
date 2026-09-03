"""Tracer-bullet end-to-end test (issue #2 acceptance criteria)."""

import dataclasses
import json
from pathlib import Path

import pytest

from tess_assoc import protocol as P
from tess_assoc.manifest import load_manifest, load_manifest_file
from tess_assoc.matcher import match
from tess_assoc.pairs import build_pairs
from tess_assoc.pipeline import render_report, run_tracer, run_tracer_dict
from tess_assoc.provider import provide_events

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tracer_v1.json"


def _manifest():
    return load_manifest_file(str(FIXTURE))


def test_manifest_rejects_sealed_sectors():
    m = _manifest()
    bad = {
        "name": m.name,
        "tic_id": m.tic_id,
        "epoch_match_tol_days": 0.3,
        "matcher_thresholds": dict(m.matcher_thresholds),
        "sectors": [{"sector": 12, "windows": [[1330.0, 1358.0]]},
                    {"sector": 93, "windows": [[2500.0, 2527.0]]}],
        "events": [],
    }
    with pytest.raises(ValueError, match="temporal leak"):
        load_manifest(bad)
    P.validate_no_temporal_leak({s.sector for s in m.sectors})


def test_manifest_rejects_coerced_types():
    m = _manifest()
    with open(FIXTURE) as f:
        good = json.load(f)
    bad_t0 = json.loads(json.dumps(good))
    bad_t0["events"][0]["t0"] = None
    with pytest.raises(ValueError):
        load_manifest(bad_t0)
    bad_sector = json.loads(json.dumps(good))
    bad_sector["events"][0]["sector"] = 12.0
    with pytest.raises(ValueError):
        load_manifest(bad_sector)


def test_pairs_unique_no_self():
    events = provide_events(_manifest())
    pairs = build_pairs(events)
    assert len(pairs) == 3  # A-B, A-C, B-C
    seen = set()
    for p in pairs:
        assert p.a_id != p.b_id
        key = tuple(sorted([p.a_id, p.b_id]))
        assert key not in seen
        seen.add(key)
    assert build_pairs({}) == []
    other_tic = dataclasses.replace(events["A"], tic_id=999)
    with pytest.raises(ValueError, match="single TIC"):
        build_pairs({"A": events["A"], "X": other_tic})


def test_matcher_compatible_only_true_repeat():
    m = _manifest()
    events = provide_events(m)
    decisions = {
        tuple(sorted([p.a_id, p.b_id])): match(
            events[p.a_id], events[p.b_id], m.matcher_thresholds
        )
        for p in build_pairs(events)
    }
    assert decisions[("A", "B")].compatible
    assert decisions[("A", "B")].explanation
    assert not decisions[("A", "C")].compatible
    assert not decisions[("B", "C")].compatible  # 5d separation: no aliases
    with pytest.raises(ValueError, match="missing key"):
        match(events["A"], events["B"], {})


def test_end_to_end_results_and_report():
    results = run_tracer_dict(json.loads(FIXTURE.read_text()))
    assert results == run_tracer(_manifest())
    assert results["sealed_sectors_touched"] == []
    assert results["protocol_version"] == "v1"
    assert len(results["pairs"]) == 3
    assert len(results["associations"]) == 1

    asc = results["associations"][0]
    assert asc["pair"] == ["A", "B"]
    assert asc["delta_t_days"] == 900.0
    assert asc["aliases_total"] == 33
    kept = {r["period_days"] for r in asc["retained"]}
    cut = {r["period_days"] for r in asc["rejected"]}
    assert 900.0 in kept and 300.0 in kept
    assert 450.0 in cut and 225.0 in cut
    assert asc["rejected"][0]["contradicting_epoch"] is not None

    json.dumps(results)  # machine-readable
    report = render_report(results)
    assert "A–B" in report and "COMPATIBLE" in report
    assert "300.0d" in report and "Sealed sectors touched: []" in report
