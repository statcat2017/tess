"""Mining-survey tests (issue #9 scale-up). Pure assembly offline first."""

import json
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc import freeze as F
from tess_assoc.learn import LearnConfig
from tess_assoc.survey import (
    build_survey_manifest,
    render_survey_report,
    run_mining_survey,
)
from tess_assoc.event import EventRecord
from tess_assoc.long_period import rank_single_transits
from tess_assoc.vetting import promote_single_transit

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REPLAY = str(FIXTURES / "replay_v1.json")
MINI = str(FIXTURES / "discovery_mini.json")

CONFIG = LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)
THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}


def _event(tic_id, sector, t0, *, snr=12.0, depth=0.01):
    return EventRecord(
        tic_id=tic_id,
        sector=sector,
        t0=t0,
        local_time=[t0 - 0.1, t0, t0 + 0.1],
        local_flux=[1.0, 1.0 - depth, 1.0],
        depth=depth,
        duration_days=0.1,
        snr=snr,
    )


def test_rank_single_transits_requires_isolation_and_long_baseline():
    windows = {29: [(2000.0, 2027.0)], 69: [(3100.0, 3127.0)]}
    candidates = rank_single_transits(
        [_event(101, 29, 2010.0), _event(102, 29, 2011.0),
         _event(102, 69, 3111.0), _event(103, 29, 2012.0, snr=6.9)],
        windows,
    )
    assert [c["tic_id"] for c in candidates] == [101]
    assert candidates[0]["period_status"] == "unconstrained"
    assert candidates[0]["baseline_days"] == pytest.approx(1127.0)


def test_rank_single_transits_rejects_short_baseline():
    event = _event(101, 29, 2010.0)
    assert rank_single_transits([event], {29: [(2000.0, 2027.0)]}, min_baseline_days=365.0) == []


def test_promote_single_transit_requires_catalog_cleanliness():
    clean = {"status": "clean"}
    low = {"status": "low"}
    small = {"status": "planetary-range"}
    verdict = promote_single_transit(
        event={"snr": 8.0}, cross_match=clean, ctoi=clean,
        contamination=low, variables=clean, companion=small,
    )
    assert verdict["candidate"] is True
    assert "period/second-transit search" in verdict["manual_checklist"][-1]
    verdict = promote_single_transit(
        event={"snr": 8.0}, cross_match={"status": "known-toi"},
        ctoi=clean, contamination=low, variables=clean, companion=small,
    )
    assert verdict["candidate"] is False


def _targets():
    return [
        {"tic_id": 1, "sectors": [12, 33]},
        {
            "tic_id": 2, "name": "TIC 2", "toi": "999.01",
            "period_days": 5.0, "t0_bjd_tdb": 2457000.0,
            "duration_hours": 2.0, "sectors": [12, 33],
        },
    ]


def test_build_survey_manifest_assembles():
    manifest = build_survey_manifest(
        "pilot", _targets(), thresholds=THRESHOLDS, purpose="mining",
        ephemeris_source="test",
    )
    assert manifest["purpose"] == "mining"
    assert [s["tic_id"] for s in manifest["systems"]] == [1, 2]
    assert manifest["systems"][0]["name"] == "TIC 1"
    assert "period_days" not in manifest["systems"][0]
    assert manifest["systems"][1]["period_days"] == 5.0
    json.dumps(manifest)
    with pytest.raises(ValueError, match="purpose"):
        build_survey_manifest("x", _targets(), thresholds=THRESHOLDS, purpose="bogus")
    with pytest.raises(ValueError, match="non-empty sectors"):
        build_survey_manifest(
            "x", [{"tic_id": 1, "sectors": []}],
            thresholds=THRESHOLDS,
        )
    with pytest.raises(ValueError, match="at least one"):
        build_survey_manifest("x", [], thresholds=THRESHOLDS)


def test_survey_resume_skips_completed(tmp_path, monkeypatch):
    import tess_assoc.survey as S

    manifest = build_survey_manifest(
        "resume", _targets(), thresholds=THRESHOLDS, purpose="rehearsal",
        ephemeris_source="test",
    )
    manifest_path = str(tmp_path / "survey.json")
    Path(manifest_path).write_text(json.dumps(manifest))
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, manifest_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    out_dir = str(tmp_path / "out")
    Path(out_dir).mkdir()
    thin = {
        "status": "complete",
        "systems_out": {"tic_id": 1, "sectors": [12, 33], "status": "complete",
                        "n_proposals": 0, "recall": {}, "pair_outcome": "not-proposed",
                        "n_cross_pairs": 0},
        "products": [],
        "pairs": [],
    }
    with open(str(Path(out_dir) / "harvest.jsonl"), "w") as f:
        f.write(json.dumps({"system": "TIC 1", **thin}) + "\n")

    calls = []

    def stub(manifest_arg, system, record=None, cache_dir=None):
        calls.append(system.name)
        assert system.name == "TIC 2"
        return {**thin, "systems_out": {**thin["systems_out"], "tic_id": 2}}

    monkeypatch.setattr(S, "harvest_system", stub)
    results = run_mining_survey(
        manifest_path, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path), out_dir=out_dir,
    )
    assert calls == ["TIC 2"]
    assert results["status"] == "complete"
    assert set(results["systems"]) == {"TIC 1", "TIC 2"}
    assert results["n_pairs_ranked"] == 0
    lines = (Path(out_dir) / "harvest.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_survey_isolates_faults(tmp_path, monkeypatch):
    import tess_assoc.survey as S

    manifest = build_survey_manifest(
        "faulty", _targets(), thresholds=THRESHOLDS, purpose="rehearsal",
        ephemeris_source="test",
    )
    manifest_path = str(tmp_path / "survey.json")
    Path(manifest_path).write_text(json.dumps(manifest))
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, manifest_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )

    def boom(manifest_arg, system, record=None, cache_dir=None):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(S, "harvest_system", boom)
    results = run_mining_survey(
        manifest_path, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path), out_dir=str(tmp_path / "out"),
    )
    assert results["status"] == "partial"
    assert results["failed_systems"] == ["TIC 1", "TIC 2"]
    assert "synthetic worker failure" in results["systems"]["TIC 1"]["reason"]
    report = render_survey_report(results)
    assert "Failed: TIC 1, TIC 2" in report
    json.dumps(results)


@needs_archive
def test_live_fetch_tois_box():
    from tess_assoc.survey import fetch_tois_box

    rows = fetch_tois_box(0, 50, -55, -25, limit=100)
    tids = {r["tic_id"] for r in rows}
    assert 166739520 in tids
    row = next(r for r in rows if r["tic_id"] == 166739520)
    assert row["toi"] == "190.01" and row["period_days"] == pytest.approx(10.02, abs=0.01)


@needs_archive
def test_live_resolve_coverage():
    from tess_assoc.survey import resolve_coverage

    coverage, failures = resolve_coverage([16740101, 42074448])
    assert failures == {}
    assert 82 in coverage[16740101]
    assert coverage[42074448] == [3, 29, 30]


@needs_archive
def test_live_cross_match_batch():
    from tess_assoc.vetting import cross_match_tois

    out = cross_match_tois([16740101, 22529346, 42074448])
    assert out["ok"]
    assert any(m["toi"] == "1150.01" for m in out["matches"][16740101])
    assert any(m["toi"] == "495.01" for m in out["matches"][22529346])
    assert out["matches"][42074448] == []


@needs_archive
def test_live_mini_survey_end_to_end(tmp_path):
    manifest = build_survey_manifest(
        "mini-live", [
            {"tic_id": 42074448, "sectors": [3, 30]},
            {"tic_id": 166739520, "name": "TOI-190.01", "toi": "190.01",
             "period_days": 10.0205932, "t0_bjd_tdb": 2460982.967362,
             "duration_hours": 5.7033765, "sectors": [30, 69]},
        ],
        thresholds=THRESHOLDS, purpose="mining", ephemeris_source="test",
    )
    manifest_path = str(tmp_path / "survey.json")
    Path(manifest_path).write_text(json.dumps(manifest))
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, manifest_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    out_dir = str(tmp_path / "out")
    first = run_mining_survey(
        manifest_path, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path), out_dir=out_dir,
        log_path=str(tmp_path / "access.jsonl"),
    )
    assert first["status"] == "complete"
    assert first["purpose"] == "mining"
    assert len(first["systems"]) == 2
    assert first["sealed_sectors_touched"] == []
    json.dumps(first)
    assert "Survey:" in render_survey_report(first)
    lines_before = (Path(out_dir) / "harvest.jsonl").read_text().strip().split("\n")
    second = run_mining_survey(
        manifest_path, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path), out_dir=out_dir,
    )
    lines_after = (Path(out_dir) / "harvest.jsonl").read_text().strip().split("\n")
    assert len(lines_after) == len(lines_before)
    assert second["n_pairs_ranked"] == first["n_pairs_ranked"]
