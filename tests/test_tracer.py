"""Tracer-bullet end-to-end test (issue #2 acceptance criteria)."""

import dataclasses
import json
from pathlib import Path

import pytest

from tess_assoc import protocol as P
from tess_assoc.manifest import load_manifest, load_manifest_file
from tess_assoc.matcher import match
from tess_assoc.pairs import build_pairs
from tess_assoc.pipeline import render_report, run_records, run_tracer, run_tracer_dict
from tess_assoc.provider import provide_events

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tracer_v1.json"
HAPPY_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "tracer_happy_v1.json"
)


def _manifest():
    return load_manifest_file(str(FIXTURE))


def _happy_manifest():
    return load_manifest_file(str(HAPPY_FIXTURE))


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


def test_manifest_rejects_unknown_and_noncanonical_structure():
    good = json.loads(FIXTURE.read_text())

    with pytest.raises(ValueError, match="unknown keys"):
        load_manifest({**good, "unexpected": True})

    bad_event = json.loads(json.dumps(good))
    bad_event["events"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="event unknown keys"):
        load_manifest(bad_event)

    bad_threshold = json.loads(json.dumps(good))
    bad_threshold["matcher_thresholds"]["unused"] = 1.0
    with pytest.raises(ValueError, match="unknown matcher thresholds"):
        load_manifest(bad_threshold)

    bad_sector = json.loads(json.dumps(good))
    bad_sector["sectors"][0]["unused"] = True
    with pytest.raises(ValueError, match="sector unknown keys"):
        load_manifest(bad_sector)

    missing_provenance = json.loads(json.dumps(good))
    del missing_provenance["events"][0]["origin"]
    with pytest.raises(ValueError, match="event missing key: origin"):
        load_manifest(missing_provenance)

    bad_windows = json.loads(json.dumps(good))
    bad_windows["sectors"][0]["windows"] = [[1340.0, 1358.0], [1330.0, 1335.0]]
    with pytest.raises(ValueError, match="sorted"):
        load_manifest(bad_windows)

    duplicate_sector = json.loads(json.dumps(good))
    duplicate_sector["sectors"].append(duplicate_sector["sectors"][0])
    with pytest.raises(ValueError, match="sector ids"):
        load_manifest(duplicate_sector)

    for key, value in (("sectors", "bad"), ("events", "bad")):
        malformed = json.loads(json.dumps(good))
        malformed[key] = value
        with pytest.raises(ValueError, match="must be a list"):
            load_manifest(malformed)


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
    assert decisions[("A", "B")].morph_corr == 1.0
    assert decisions[("A", "C")].morph_corr == decisions[("B", "C")].morph_corr
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


def test_happy_path_is_reproducible_without_contradictions():
    manifest = _happy_manifest()
    first = run_tracer(manifest)
    second = run_tracer(manifest)

    assert first == second
    assert first["sealed_sectors_touched"] == []
    assert len(first["events"]) == 2
    assert len(first["pairs"]) == 1
    assert first["pairs"][0]["compatible"] is True
    assert len(first["associations"]) == 1
    association = first["associations"][0]
    assert association["rejected"] == []
    assert association["retained"]


def test_programmatic_tracer_rejects_sealed_sectors_before_processing():
    manifest = _happy_manifest()
    bad = dataclasses.replace(
        manifest,
        sectors=manifest.sectors
        + (dataclasses.replace(manifest.sectors[0], sector=80),),
    )
    with pytest.raises(ValueError, match="temporal leak"):
        run_tracer(bad)


def test_records_reject_sealed_event_records_before_processing():
    manifest = _happy_manifest()
    events = provide_events(manifest)
    sealed = dataclasses.replace(events["A"], sector=80)
    events["A"] = sealed
    with pytest.raises(ValueError, match="temporal leak"):
        run_records(manifest, events)
