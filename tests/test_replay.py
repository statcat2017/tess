"""Known-system replay tests (issue #3 acceptance criteria).

Pure-unit tests always run. Archive/network tests skip cleanly when
dependencies or MAST are unreachable — and report unavailability clearly.
"""

import json
import os
import dataclasses
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc.archive import ArchiveProduct, ArchiveUnavailable, cache_dir, find_spoc_ffi_uri
from tess_assoc.extract import (
    BTJD_OFFSET,
    extract_events,
    predicted_transits,
)
from tess_assoc.manifest import ReplaySystem
from tess_assoc.observability import CadenceEvidence, LightCurve, coverage_windows
from tess_assoc.replay import MISS_REASONS, load_replay_manifest, replay_all, replay_system

REPLAY = Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_predicted_transits_enumeration():
    assert predicted_transits(2457000.0, 1.0, 0.0, 3.0) == [0.0, 1.0, 2.0, 3.0]
    assert predicted_transits(2457000.5, 1.0, 0.0, 3.0) == [0.5, 1.5, 2.5]
    assert BTJD_OFFSET == 2457000.0
    with pytest.raises(ValueError):
        predicted_transits(2457000.0, 0.0, 0.0, 3.0)


def test_coverage_windows_split_on_gaps():
    assert coverage_windows([]) == []
    assert coverage_windows([1.0, 1.1, 1.2, 5.0, 5.1]) == [(1.0, 1.2), (5.0, 5.1)]
    assert coverage_windows([1.0, 1.1, 5.0]) == [(1.0, 1.1)]
    assert coverage_windows([7.0]) == []


def test_coverage_windows_split_short_quality_gap():
    time = [i * 0.02 for i in range(11)] + [0.42 + i * 0.02 for i in range(11)]
    assert coverage_windows(time) == [(0.0, 0.2), (0.42, 0.62)]


def test_cadence_evidence_preserves_quality_gaps_and_roundtrips():
    evidence = CadenceEvidence(
        time=(0.0, 0.1, 0.2, 0.5, 0.6),
        usable=(True, True, False, True, True),
        quality_flags=(0, 0, 4, 0, 0),
    )
    assert evidence.observing_windows == ((0.0, 0.1), (0.5, 0.6))
    assert evidence.missing == 1
    assert evidence.quality_flagged == 1
    assert CadenceEvidence.from_dict(evidence.to_dict()) == evidence


def test_cadence_evidence_rejects_misaligned_or_false_windows():
    with pytest.raises(ValueError):
        CadenceEvidence((0.0, 0.1), (True,), (0, 0))
    with pytest.raises(ValueError):
        CadenceEvidence(
            (0.0, 0.1), (True, False), (0, 4), observing_windows=((0.0, 0.1),)
        )


def test_coverage_windows_respects_quality_mask_at_sector_boundaries():
    assert coverage_windows(
        [10.0, 10.1, 10.2, 20.0, 20.1], usable=[True, True, False, True, True]
    ) == [(10.0, 10.1), (20.0, 20.1)]


def test_all_unusable_lightcurve_retains_cadence_evidence():
    evidence = CadenceEvidence(
        time=(0.0, 0.1), usable=(False, False), quality_flags=(4, 4)
    )
    curve = LightCurve(time=(), flux=(), evidence=evidence)
    assert curve.evidence.missing == 2
    assert curve.time == ()


def test_extraction_skips_transit_inside_short_quality_gap(monkeypatch):
    import tess_assoc.extract as E

    time = [-1.0 + i * 0.02 for i in range(61)] + [0.42 + i * 0.02 for i in range(30)]
    flux = [0.99 if abs(t - 0.1) <= 0.04 else 1.0 for t in time]
    evidence = CadenceEvidence(
        time=tuple(time),
        usable=(True,) * len(time),
        quality_flags=(0,) * len(time),
    )
    monkeypatch.setattr(
        E,
        "load_lightcurve",
        lambda product: LightCurve(tuple(time), tuple(flux), evidence),
    )
    monkeypatch.setattr(E, "refine_epoch", lambda *args, **kwargs: 0.1)
    product = ArchiveProduct(1, 12, "unused", "unused", "now", True)
    system = ReplaySystem(
        name="gap", tic_id=1, period_days=1.0,
        t0_bjd_tdb=BTJD_OFFSET + 0.1, duration_hours=2.0, sectors=[12],
    )
    extracted, skipped, _ = extract_events(product, system)
    assert extracted == []
    assert skipped[0].reason == "insufficient full observing window coverage"


def test_epoch_refinement_can_move_edge_prediction_into_full_coverage(monkeypatch):
    import tess_assoc.extract as E

    time = [BTJD_OFFSET + i * 0.02 for i in range(501)]
    flux = [0.98 if abs(t - (BTJD_OFFSET + 1.0)) <= 0.08 else 1.0 for t in time]
    evidence = CadenceEvidence(
        time=tuple(time),
        usable=(True,) * len(time),
        quality_flags=(0,) * len(time),
    )
    monkeypatch.setattr(
        E,
        "load_lightcurve",
        lambda product: LightCurve(tuple(time), tuple(flux), evidence),
    )
    product = ArchiveProduct(1, 12, "unused", "unused", "now", True)
    system = ReplaySystem(
        name="edge", tic_id=1, period_days=20.0,
        t0_bjd_tdb=BTJD_OFFSET + 0.4, duration_hours=4.8, sectors=[12],
    )
    extracted, skipped, _ = extract_events(product, system)
    assert len(extracted) == 1
    assert not skipped
    assert abs(extracted[0].record.t0 - (BTJD_OFFSET + 1.0)) < 0.1


def test_coverage_windows_split_on_known_transit_masks():
    assert coverage_windows(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        excluded_windows=[(0.2, 0.3)],
    ) == [(0.0, 0.2), (0.3, 0.5)]


def test_cache_dir_outside_repo_and_env_override(tmp_path):
    assert not os.path.realpath(cache_dir()).startswith(os.path.realpath(REPO_ROOT))
    os.environ["TESS_ASSOC_CACHE"] = str(tmp_path)
    try:
        assert cache_dir() == str(tmp_path)
    finally:
        del os.environ["TESS_ASSOC_CACHE"]


def test_replay_manifest_declares_spoc_product():
    replay = load_replay_manifest(str(REPLAY))
    assert replay.product == "TESS-SPOC FFI"
    assert 3 <= len(replay.systems) <= 5
    for system in replay.systems:
        assert len(system.sectors) >= 2


def test_replay_manifest_rejects_invalid_matcher_thresholds():
    replay = load_replay_manifest(str(REPLAY))
    for key, value in (
        ("max_rel_depth_diff", -1.0),
        ("max_rel_duration_diff", -1.0),
        ("min_morph_corr", 1.1),
    ):
        thresholds = dict(replay.matcher_thresholds)
        thresholds[key] = value
        with pytest.raises(ValueError, match="threshold"):
            dataclasses.replace(replay, matcher_thresholds=thresholds)


def test_replay_manifest_rejects_duplicate_system_identity():
    replay = load_replay_manifest(str(REPLAY))
    with pytest.raises(ValueError, match="system names"):
        dataclasses.replace(replay, systems=(replay.systems[0], replay.systems[0]))
    duplicate_tic = dataclasses.replace(
        replay.systems[1], name="duplicate name", tic_id=replay.systems[0].tic_id
    )
    with pytest.raises(ValueError, match="TIC ids"):
        dataclasses.replace(replay, systems=(replay.systems[0], duplicate_tic))


def test_replay_manifest_rejects_wrong_product(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError, match="missing key"):
        load_replay_manifest(str(bad))


@needs_archive
def test_archive_reports_clearly_for_unknown_target():
    with pytest.raises(ArchiveUnavailable):
        find_spoc_ffi_uri(99999999999, 1)


@needs_archive
def test_live_blind_replay_measures_recall(tmp_path):
    from tess_assoc.replay import load_replay_manifest, replay_blind_system

    replay = load_replay_manifest(str(REPLAY))
    system = next(s for s in replay.systems if s.name == "WASP-121 b")
    res = replay_blind_system(replay, system, cache_dir=str(tmp_path))
    assert res["sealed_sectors_touched"] == []
    assert res["n_proposals"] > 0
    assert res["recall"]["rate"] >= 0.8, res["recall"]
    assert res["pair_outcome"] == "associated", res["pair_outcome"]
    missed = res["missed"]
    assert len(missed) == res["recall"]["known"] - res["recall"]["recalled"]
    for m in missed:
        assert set(m) == {
            "sector", "t0", "max_snr", "proposed", "reason", "observability"
        }
        assert m["reason"] in MISS_REASONS
        assert m["max_snr"] is None or isinstance(m["max_snr"], float)
    assert res["recall"]["coverable"] <= res["recall"]["known"]
    assert res["recall"]["recalled_coverable"] <= res["recall"]["coverable"]
    json.dumps(res)


@needs_archive
def test_live_replay_three_systems(tmp_path):
    from tess_assoc.orbit import generate_aliases

    results = replay_all(str(REPLAY), cache_dir=str(tmp_path))
    assert set(results) == {"WASP-43 b", "WASP-121 b", "KELT-9 b"}
    for name, res in results.items():
        # Known true pairs associate on real morphology.
        assert res["sealed_sectors_touched"] == []
        assert len(res["anchors"]) == 2
        pair_map = {
            tuple(sorted([p["a"], p["b"]])): p for p in res["pairs"]
        }
        assert pair_map[tuple(sorted(res["anchors"]))]["compatible"]
        # Deterministic alias/window machinery on the anchor pair.
        assoc = [a for a in res["associations"] if sorted(a["pair"]) == sorted(res["anchors"])]
        assert assoc, f"{name}: anchor pair has no association"
        assert assoc[0]["aliases_total"] == len(generate_aliases(assoc[0]["delta_t_days"]))
        assert len(assoc[0]["retained"]) + len(assoc[0]["rejected"]) == assoc[0]["aliases_total"]
        for prod in res["products"]:
            assert prod["data_uri"].endswith("_lc.fits")
            assert not os.path.realpath(prod["local_path"]).startswith(
                os.path.realpath(REPO_ROOT)
            )
        json.dumps(res)

    # Science outputs are deterministic across runs (provenance timestamps excluded).
    replay = load_replay_manifest(str(REPLAY))
    again = replay_system(replay, replay.systems[0], cache_dir=str(tmp_path))
    first = results[replay.systems[0].name]
    assert again["pairs"] == first["pairs"]
    assert again["associations"] == first["associations"]
