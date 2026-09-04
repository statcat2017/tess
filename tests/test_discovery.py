"""Discovery cohort tests (issue #9). Offline gate/vetting first."""

import json
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc import freeze as F
from tess_assoc.discovery import (
    DiscoveryManifest,
    DiscoverySystem,
    load_discovery_manifest,
    render_discovery_report,
)
from tess_assoc.event import EventRecord
from tess_assoc.learn import LearnConfig
from tess_assoc.vetting import (
    MANUAL_CHECKLIST,
    check_contamination,
    cross_match_toi,
    promote_candidate,
    secondary_search,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REPLAY = str(FIXTURES / "replay_v1.json")
MINI = str(FIXTURES / "discovery_mini.json")
COHORT = str(FIXTURES / "discovery_v1.json")

CONFIG = LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)
THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}


def _rec(tic, sector, t0, depth=0.01):
    n = 21
    times = [t0 - 0.3 + i * 0.03 for i in range(n)]
    fluxes = [1.0 - depth if abs(t - t0) <= 0.1 else 1.0 for t in times]
    return EventRecord(
        tic_id=tic, sector=sector, t0=t0, local_time=times, local_flux=fluxes,
        depth=depth, duration_days=0.2, snr=10.0,
        stellar_meta={}, quality={},
    )


def _freeze(tmp_path, manifest=MINI):
    path = str(tmp_path / "discovery_freeze.json")
    record = F.create_freeze(
        REPLAY, manifest, CONFIG, output_path=path, cohort_key="discovery"
    )
    assert record.manifests["discovery"]["sha256"] == F.file_hash(manifest)
    return path, record


def test_discovery_structs_reject_sealed_but_allow_106():
    with pytest.raises(ValueError, match="not allowed"):
        DiscoverySystem(name="x", tic_id=1, sectors=[82])
    with pytest.raises(ValueError, match="not allowed"):
        DiscoverySystem(name="x", tic_id=1, sectors=[80])
    ok = DiscoverySystem(name="x", tic_id=1, sectors=[12, 106])
    assert ok.sectors == (12, 106)
    with pytest.raises(ValueError, match="non-empty list"):
        DiscoveryManifest(
            name="x", product="TESS-SPOC FFI", ephemeris_source="e",
            epoch_match_tol_days=0.3, window_half_span_days=0.6,
            resample_samples=61,
            matcher_thresholds={
                "max_rel_depth_diff": 0.25,
                "max_rel_duration_diff": 0.25,
                "min_morph_corr": 0.9,
            },
            systems=(),
        )


def test_dev_loaders_reject_discovery_manifest():
    from tess_assoc.replay import load_replay_manifest

    with pytest.raises(ValueError, match="temporal leak"):
        load_replay_manifest(MINI)
    with pytest.raises(ValueError, match="temporal leak"):
        load_replay_manifest(COHORT)


def test_discovery_gate_needs_valid_freeze(tmp_path):
    path, _ = _freeze(tmp_path)
    manifest = load_discovery_manifest(MINI, path, CONFIG)
    assert [s.sectors for s in manifest.systems] == [(12, 106)]
    import dataclasses

    stale = F.load_freeze_record(path)
    stale = dataclasses.replace(stale, code_sha="0" * 64)
    with pytest.raises(ValueError, match="source tree changed"):
        load_discovery_manifest(MINI, stale, CONFIG)


def test_combine_secondary_searches_unions_both_sectors():
    from tess_assoc.vetting import combine_secondary_searches

    flat = {"aliases": [{"period_days": 10.0, "secondary_found": False, "max_snr": None}]}
    assert combine_secondary_searches([10.0], [flat, flat])["n_flagged"] == 0
    hit = {"aliases": [{"period_days": 10.0, "secondary_found": True, "max_snr": 6.0}]}
    merged = combine_secondary_searches([10.0, 20.0], [flat, hit])
    assert merged["n_flagged"] == 1
    assert merged["flagged_periods"] == [10.0]
    assert merged["worst"] == {"period_days": 10.0, "max_snr": 6.0}


def test_secondary_in_second_sector_convicts():
    import random

    from tess_assoc.propose import detrend
    from tess_assoc.vetting import combine_secondary_searches, secondary_search

    rng = random.Random(9)
    time_a = [100.0 + i * 0.02 for i in range(500)]
    flat_a = [1.0 + rng.gauss(0, 0.001) for _ in time_a]
    time_b = [200.0 + i * 0.02 for i in range(500)]
    flat_b = [1.0 + rng.gauss(0, 0.001) for _ in time_b]
    dipped_b = [f - 0.02 if abs(t - 206.0) <= 0.1 else f for t, f in zip(time_b, flat_b)]
    det_a, sig_a = detrend(time_a, flat_a)
    det_b, sig_b = detrend(time_b, dipped_b)
    res_a = secondary_search(time_a, det_a, sig_a, 100.0, [10.0], 0.1)
    res_b = secondary_search(time_b, det_b, sig_b, 201.0, [10.0], 0.1)
    assert res_a["n_flagged"] == 0
    merged = combine_secondary_searches([10.0], [res_a, res_b])
    assert merged["n_flagged"] == 1


def test_discovery_manifest_rejects_duplicate_names():
    from tess_assoc.discovery import DiscoveryManifest, DiscoverySystem

    base = dict(
        name="x", product="TESS-SPOC FFI", ephemeris_source="e",
        epoch_match_tol_days=0.3, window_half_span_days=0.6,
        resample_samples=61,
        matcher_thresholds={
            "max_rel_depth_diff": 0.25,
            "max_rel_duration_diff": 0.25,
            "min_morph_corr": 0.9,
        },
    )
    dup = DiscoverySystem(name="same", tic_id=1, sectors=[12])
    with pytest.raises(ValueError, match="unique"):
        DiscoveryManifest(
            systems=(dup, DiscoverySystem(name="same", tic_id=2, sectors=[12])),
            **base,
        )


def test_cross_epoch_close_pair_skips_aliases():
    from tess_assoc.discovery import _cross_epoch_pairs

    recs = {
        "a": _rec(1, 12, 100.0),
        "b": _rec(1, 39, 120.0),
        "c": _rec(1, 39, 1000.0),
    }
    pairs = _cross_epoch_pairs(
        recs, {12: [(90.0, 120.0)], 39: [(110.0, 130.0), (990.0, 1020.0)]},
        THRESHOLDS,
    )
    assert pairs, "expected cross-sector pairs"
    close = [p for p in pairs if abs(p["event_b"]["t0"] - 120.0) < 1.0]
    assert close and all(
        p["retained_periods"] == [] and p["aliases_total"] == 0 for p in close
    )
    far = [p for p in pairs if abs(p["event_b"]["t0"] - 1000.0) < 1.0]
    assert far and all(p["aliases_total"] > 0 for p in far)


def test_triage_caps_pairs_per_system():
    from tess_assoc.discovery import DiscoveryManifest, DiscoverySystem, triage_ranked_pairs

    def pair(t0, score, compatible=True):
        return {
            "a": "x", "b": "y",
            "event_a": {"sector": 12, "t0": t0, "depth": 0.01,
                        "duration_days": 0.2, "snr": 10.0},
            "event_b": {"sector": 39, "t0": t0 + 900.0, "depth": 0.01,
                        "duration_days": 0.2, "snr": 10.0},
            "compatible": compatible, "score": score, "morph_corr": 0.99,
            "retained_periods": [900.0, 450.0], "aliases_total": 33,
        }

    manifest = DiscoveryManifest(
        name="cap", product="TESS-SPOC FFI", ephemeris_source="e",
        epoch_match_tol_days=0.3, window_half_span_days=0.6,
        resample_samples=61, matcher_thresholds=dict(THRESHOLDS),
        systems=(
            DiscoverySystem(name="A", tic_id=1, sectors=[12, 39]),
            DiscoverySystem(name="B", tic_id=2, sectors=[12, 39]),
        ),
        purpose="mining",
    )
    harvests = {
        "A": {"pairs": [pair(100.0 + i, 0.9 - i * 0.01) for i in range(8)],
              "products": []},
        "B": {"pairs": [pair(200.0 + i, 0.8 - i * 0.01) for i in range(2)],
              "products": []},
    }
    clean = {
        1: {"contamination": {"status": "low"}, "cross_match": {"status": "clean"}},
        2: {"contamination": {"status": "low"}, "cross_match": {"status": "clean"}},
    }
    candidates, reviewed = triage_ranked_pairs(
        manifest, harvests, shortlist_k=10, per_system_cap=2,
        catalog_prefetch=clean,
    )
    assert len(candidates) == 4
    assert [c["system"] for c in candidates] == ["A", "A", "B", "B"]


def test_secondary_search_flags_real_dip_and_clears_flat():
    import random

    rng = random.Random(4)
    time = [200.0 + i * 0.02 for i in range(500)]
    flat = [1.0 + rng.gauss(0, 0.001) for _ in time]
    from tess_assoc.propose import detrend

    detrended, sigma = detrend(time, flat)
    clean = secondary_search(time, detrended, sigma, 200.0, [10.0], 0.1)
    assert clean["n_flagged"] == 0 and clean["worst"] is None
    dipped = [
        f - 0.02 if abs(t - 205.0) <= 0.1 else f for t, f in zip(time, flat)
    ]
    detrended2, sigma2 = detrend(time, dipped)
    hit = secondary_search(time, detrended2, sigma2, 200.0, [10.0], 0.1)
    assert hit["n_flagged"] == 1
    assert hit["worst"]["period_days"] == 10.0
    assert hit["worst"]["max_snr"] >= 4.0
    with pytest.raises(ValueError, match="alias period"):
        secondary_search(time, detrended, sigma, 200.0, [0.0], 0.1)


def test_contamination_and_toi_rules():
    low = check_contamination(1, contratio=0.002)
    assert low["status"] == "low"
    high = check_contamination(1, contratio=0.5)
    assert high["status"] == "high" and "pixel" in high["reason"]
    clean = cross_match_toi(1, toi_rows=[])
    assert clean["status"] == "clean"
    known = cross_match_toi(1, toi_rows=[{"toi": "1.01", "disposition": "KP"}])
    assert known["status"] == "known-toi"
    assert MANUAL_CHECKLIST


def test_companion_radius_blocks_stellar():
    from tess_assoc.vetting import check_companion_radius

    eb = check_companion_radius(1, 0.053, rad=1.64)
    assert eb["status"] == "stellar"
    assert eb["companion_r_sun"] == pytest.approx(0.378, abs=0.01)
    planet = check_companion_radius(1, 0.01, rad=1.0)
    assert planet["status"] == "planetary-range"
    base = dict(
        compatible=True, aliases_retained=3,
        cross_match={"status": "clean"},
        contamination={"status": "low"}, secondary={"n_flagged": 0},
    )
    assert promote_candidate(**base, companion=planet)["candidate"]
    blocked = promote_candidate(**base, companion=eb)
    assert not blocked["candidate"]
    assert any("stellar" in r for r in blocked["reasons"])
    unknown = promote_candidate(
        **base, companion={"status": "unknown", "reason": "no radius"}
    )
    assert not unknown["candidate"]


def test_variables_gate_promotion():
    from tess_assoc.vetting import check_variables

    base = dict(
        compatible=True, aliases_retained=3,
        cross_match={"status": "clean"},
        contamination={"status": "low"}, secondary={"n_flagged": 0},
        companion={"status": "planetary-range"},
    )
    assert promote_candidate(
        **base, variables={"status": "clean"}
    )["candidate"]
    blocked = promote_candidate(
        **base, variables={"status": "known-variable", "matches": [{"catalog": "asassn"}]}
    )
    assert not blocked["candidate"]
    assert any("variable" in r for r in blocked["reasons"])
    assert not promote_candidate(
        **base, variables={"status": "unknown"}
    )["candidate"]


def test_promotion_requires_everything_clean():
    cross = {"status": "clean"}
    contam = {"status": "low"}
    sec = {"n_flagged": 0}
    good = promote_candidate(
        compatible=True, aliases_retained=3, cross_match=cross,
        contamination=contam, secondary=sec,
    )
    assert good["candidate"] and not good["reasons"]
    assert good["manual_checklist"] == list(MANUAL_CHECKLIST)
    bad = promote_candidate(
        compatible=True, aliases_retained=3,
        cross_match={"status": "known-toi"},
        contamination=contam, secondary=sec,
    )
    assert not bad["candidate"] and any("TOI" in r for r in bad["reasons"])
    eb = promote_candidate(
        compatible=True, aliases_retained=2, cross_match=cross,
        contamination=contam,
        secondary={"n_flagged": 1},
    )
    assert not eb["candidate"]
    no_alias = promote_candidate(
        compatible=True, aliases_retained=0, cross_match=cross,
        contamination=contam, secondary=sec,
    )
    assert not no_alias["candidate"]


def test_discovery_blocked_and_partial_statuses(tmp_path, monkeypatch):
    import copy

    import tess_assoc.discovery as D
    from tess_assoc.archive import ArchiveUnavailable

    manifest_dict = json.loads(Path(MINI).read_text())
    manifest_dict["systems"].append(
        {
            "name": "Dev b",
            "tic_id": 2,
            "period_days": 5.0,
            "t0_bjd_tdb": 2457000.0,
            "duration_hours": 2.0,
            "sectors": [12, 33],
        }
    )
    manifest_path = str(tmp_path / "mixed.json")
    Path(manifest_path).write_text(json.dumps(manifest_dict))
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, manifest_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    manifest = load_discovery_manifest(manifest_path, freeze_path, CONFIG)

    def fake_empty_result():
        return {
            "tic_id": 2,
            "events": [],
            "sectors": [
                {"sector": 12, "windows": [[100.0, 120.0]]},
                {"sector": 33, "windows": [[200.0, 220.0]]},
            ],
            "products": [],
            "n_proposals": 0,
            "recall": {
                "known": 0, "recalled": 0, "rate": 0.0, "coverable": 0,
                "recalled_coverable": 0, "rate_coverable": 0.0,
            },
            "pair_outcome": "not-proposed",
            "sealed_sectors_touched": [],
        }

    def stub(manifest_arg, system, cache_dir=None, records_runner=None):
        if system.tic_id == 99999999:
            raise ArchiveUnavailable("no Sector 106 SPOC yet")
        return copy.deepcopy(fake_empty_result())

    monkeypatch.setattr(D, "replay_blind_system", stub)
    results = D.run_discovery(
        manifest, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path),
    )
    assert results["status"] == "partial"
    assert results["blocked_systems"] == ["Mini b"]
    assert results["systems"]["Mini b"]["status"] == "blocked-on-archive"
    assert results["systems"]["Dev b"]["status"] == "complete"
    assert results["sealed_sectors_touched"] == []
    assert results["candidates"] == [] and results["reviewed"] == []


def test_discovery_report_language():
    results = {
        "protocol_version": "v1",
        "cohort": "discovery-106",
        "is_discovery": True,
        "status": "complete",
        "freeze": {"code_sha": "abc123", "created_utc": "t", "unblinded_utc": "u"},
        "sealed_sectors_touched": [],
        "systems": {
            "TOI-1.01": {
                "sectors": [12, 106], "n_proposals": 10, "n_cross_pairs": 4,
            }
        },
        "blocked_systems": [],
        "n_pairs_ranked": 4,
        "candidates": [],
        "reviewed": [
            {
                "tic_id": 1,
                "promotion": {"candidate": False, "reasons": ["TOI cross-match: known-toi"]},
            }
        ],
    }
    report = render_discovery_report(results)
    assert "NOT confirmed planets" in report
    assert "known-toi" in report
    json.dumps({k: v for k, v in results.items()})


@needs_archive
def test_live_cohort_selection_probes_old_footprint():
    from tess_assoc.discovery import sectors_for_tic, select_cohort

    assert 82 in sectors_for_tic(16740101)
    machinery = select_cohort(
        25.0148, -40.1221, 30, radius_deg=0.5, mag_limit=12.5, max_targets=5,
    )
    assert machinery, "cohort selection found nothing in a populated footprint"
    assert all(30 in c["sectors"] and c["early_sectors"] for c in machinery)
    empty = select_cohort(
        25.0148, -40.1221, 106, radius_deg=0.5, mag_limit=12.5, max_targets=5,
    )
    assert empty == [], "Sector 106 SPOC unexpectedly archived"


@needs_archive
def test_live_mining_validation_excludes_known_toi(tmp_path):
    from tess_assoc.discovery import run_discovery

    validation_path = str(FIXTURES / "mining_validation.json")
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, validation_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    manifest = load_discovery_manifest(validation_path, freeze_path, CONFIG)
    results = run_discovery(
        manifest, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path),
    )
    assert results["purpose"] == "mining"
    assert results["status"] == "complete"
    assert results["sealed_sectors_touched"] == []
    assert results["n_pairs_ranked"] >= 1
    assert results["candidates"] == []
    assert results["reviewed"]
    assert all(
        any("known-toi" in r for r in e["promotion"]["reasons"])
        for e in results["reviewed"]
    )
    json.dumps(results)
    assert "NOT confirmed planets" in render_discovery_report(results)


@needs_archive
def test_live_mining_hunt_reports_cleanly(tmp_path):
    from tess_assoc.discovery import run_discovery

    hunt_path = str(FIXTURES / "mining_v1.json")
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, hunt_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    manifest = load_discovery_manifest(hunt_path, freeze_path, CONFIG)
    assert len(manifest.systems) == 8
    results = run_discovery(
        manifest, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path),
    )
    assert results["purpose"] == "mining"
    assert results["status"] == "complete"
    assert results["sealed_sectors_touched"] == []
    assert len(results["systems"]) == 8
    for entry in results["candidates"]:
        assert entry["promotion"]["candidate"]
        assert entry["vetting"]["cross_match"]["status"] == "clean"
    json.dumps(results)
    report = render_discovery_report(results)
    assert "NOT confirmed planets" in report


@needs_archive
def test_live_106_blocked_on_archive(tmp_path):
    from tess_assoc.archive import ArchiveUnavailable, find_spoc_ffi_uri

    with pytest.raises(ArchiveUnavailable):
        find_spoc_ffi_uri(166739520, 106)


@needs_archive
def test_live_rehearsal_run_on_dev(tmp_path):
    from tess_assoc.discovery import run_discovery

    rehearsal = {
        "name": "rehearsal-dev",
        "product": "TESS-SPOC FFI",
        "ephemeris_source": "rehearsal (dev data as discovery stand-in)",
        "epoch_match_tol_days": 0.3,
        "window_half_span_days": 0.6,
        "resample_samples": 61,
        "matcher_thresholds": {
            "max_rel_depth_diff": 0.25,
            "max_rel_duration_diff": 0.25,
            "min_morph_corr": 0.9,
        },
        "systems": [
            {
                "name": "WASP-121 b",
                "tic_id": 22529346,
                "period_days": 1.27492504,
                "t0_bjd_tdb": 2460245.02038,
                "duration_hours": 2.9053,
                "sectors": [7, 33],
            }
        ],
    }
    manifest_path = str(tmp_path / "rehearsal.json")
    Path(manifest_path).write_text(json.dumps(rehearsal))
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, manifest_path, CONFIG, output_path=freeze_path,
        cohort_key="discovery",
    )
    manifest = load_discovery_manifest(manifest_path, freeze_path, CONFIG)
    results = run_discovery(
        manifest, freeze_path=freeze_path, config=CONFIG,
        cache_dir=str(tmp_path), log_path=str(tmp_path / "access.jsonl"),
    )
    assert results["is_discovery"] is False
    assert results["status"] == "complete"
    assert results["sealed_sectors_touched"] == []
    assert results["candidates"] == []
    assert results["n_pairs_ranked"] >= 1
    assert results["reviewed"]
    assert all(
        "rehearsal" in "; ".join(r["promotion"]["reasons"])
        for r in results["reviewed"][:1]
    )
    json.dumps(results)
    report = render_discovery_report(results)
    assert "NOT confirmed planets" in report
    log_lines = (tmp_path / "access.jsonl").read_text().strip().split("\n")
    assert json.loads(log_lines[0])["event"] == "discovery_run"


@needs_archive
def test_live_variable_catalog_statuses():
    from tess_assoc.vetting import check_variables

    eb2 = check_variables(197931848)
    assert eb2["status"] == "known-variable"
    assert any(m["catalog"] == "asassn" for m in eb2["matches"])
    asassn = [m for m in eb2["matches"] if m["catalog"] == "asassn"][0]
    assert asassn["type"] == "EA"
    eb1 = check_variables(224224413)
    assert eb1["status"] == "clean"


@needs_archive
def test_live_vetting_excludes_known_toi():
    contam = check_contamination(16740101)
    assert contam["status"] == "low"
    match = cross_match_toi(16740101)
    assert match["status"] == "known-toi"
    assert any(m["toi"] == "1150.01" for m in match["matches"])
    verdict = promote_candidate(
        compatible=True, aliases_retained=5, cross_match=match,
        contamination=contam, secondary={"n_flagged": 0},
    )
    assert not verdict["candidate"]
