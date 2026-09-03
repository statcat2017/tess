"""Freeze + temporal-holdout tests (issue #8). Offline gate mechanics first."""

import dataclasses
import json
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc import freeze as F
from tess_assoc.event import EventRecord
from tess_assoc.holdout import render_holdout_report
from tess_assoc.learn import LearnConfig
from tess_assoc.learn import test_metrics as compute_metrics
from tess_assoc.manifest import ManifestSector, TracerManifest
from tess_assoc.pipeline import run_holdout_records, run_records

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REPLAY = str(FIXTURES / "replay_v1.json")
HOLDOUT = str(FIXTURES / "holdout_v1.json")
MINI = str(FIXTURES / "holdout_mini.json")

THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}
CONFIG = LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)


def _rec(tic, sector, t0, depth=0.01):
    n = 21
    times = [t0 - 0.3 + i * 0.03 for i in range(n)]
    fluxes = [1.0 - depth if abs(t - t0) <= 0.1 else 1.0 for t in times]
    return EventRecord(
        tic_id=tic, sector=sector, t0=t0, local_time=times, local_flux=fluxes,
        depth=depth, duration_days=0.2, snr=10.0,
        stellar_meta={}, quality={},
    )


def _manifest(sectors=(12, 80)):
    return TracerManifest(
        name="mini", tic_id=1, epoch_match_tol_days=0.3,
        matcher_thresholds=dict(THRESHOLDS),
        sectors=tuple(
            ManifestSector(sector=s, windows=((100.0, 120.0),)) for s in sectors
        ),
        events=(),
    )


def _freeze(tmp_path, **overrides):
    path = str(tmp_path / "freeze.json")
    record = F.create_freeze(REPLAY, MINI, CONFIG, output_path=path, **overrides)
    return path, record


def test_freeze_round_trip_and_verify(tmp_path):
    path, record = _freeze(tmp_path)
    assert record.unblinded_utc is None
    assert record.protocol_version == "v1"
    assert record.systems["holdout"] == [99999999]
    same = F.verify_freeze(path, CONFIG)
    assert same.code_sha == record.code_sha
    json.dumps(record.to_dict())
    assert F.FreezeRecord.from_dict(record.to_dict()) == record


def test_freeze_rejects_tampering(tmp_path):
    path, record = _freeze(tmp_path)
    bad_sha = dataclasses.replace(record, code_sha="0" * 64)
    with pytest.raises(ValueError, match="source tree changed"):
        F.verify_freeze(bad_sha, CONFIG)
    bad_thresholds = dataclasses.replace(
        record, thresholds={**record.thresholds, "min_morph_corr": 0.5}
    )
    with pytest.raises(ValueError, match="thresholds"):
        F.verify_freeze(bad_thresholds, CONFIG)
    with pytest.raises(ValueError, match="learn config"):
        F.verify_freeze(path, dataclasses.replace(CONFIG, seed=99))
    with pytest.raises(ValueError, match="missing key"):
        F.FreezeRecord.from_dict({"protocol_version": "v1"})


def test_dev_loaders_reject_sealed_holdout_manifests():
    from tess_assoc.replay import load_replay_manifest

    for path in (HOLDOUT, MINI):
        with pytest.raises(ValueError, match="temporal leak"):
            load_replay_manifest(path)


def test_gate_requires_valid_freeze(tmp_path):
    with pytest.raises(OSError):
        F.load_holdout_manifest(MINI, str(tmp_path / "missing.json"), CONFIG)
    path, record = _freeze(tmp_path)
    with pytest.raises(ValueError, match="mismatch|changed|differ|covered|failed"):
        F.load_holdout_manifest(
            MINI, dataclasses.replace(record, code_sha="0" * 64), CONFIG
        )
    manifest = F.load_holdout_manifest(MINI, path, CONFIG)
    assert [s.sectors for s in manifest.systems] == [(12, 80)]
    assert manifest.product == "TESS-SPOC FFI"


def test_gate_binds_on_bytes_not_location(tmp_path):
    import shutil

    freeze_path, _ = _freeze(tmp_path)
    relocated = str(tmp_path / "renamed_holdout.json")
    shutil.copy(MINI, relocated)
    manifest = F.load_holdout_manifest(relocated, freeze_path, CONFIG)
    assert manifest.name == "holdout_mini"
    with open(relocated, "a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="bytes differ"):
        F.load_holdout_manifest(relocated, freeze_path, CONFIG)


def test_mark_unblinded_stamps_once(tmp_path):
    path, _ = _freeze(tmp_path)
    first = F.mark_unblinded(path)
    assert first.unblinded_utc is not None
    second = F.mark_unblinded(path)
    assert second.unblinded_utc == first.unblinded_utc


def test_holdout_records_need_freeze_but_run_sealed(tmp_path):
    manifest = _manifest()
    events = {"a": _rec(1, 12, 100.0), "b": _rec(1, 80, 200.0)}
    with pytest.raises(ValueError, match="temporal leak"):
        run_records(manifest, events)
    path, record = _freeze(tmp_path)
    out = run_holdout_records(manifest, events, freeze_record=record)
    assert out["sealed_sectors_touched"] == [80]
    assert out["freeze"]["code_sha"] == record.code_sha
    assert len(out["pairs"]) == 1
    json.dumps(out)
    stale = dataclasses.replace(record, code_sha="0" * 64)
    with pytest.raises(ValueError, match="changed since freeze"):
        run_holdout_records(manifest, events, freeze_record=stale)
    drifted = _manifest()
    object.__setattr__(
        drifted, "matcher_thresholds", {**THRESHOLDS, "min_morph_corr": 0.5}
    )
    with pytest.raises(ValueError, match="differ from frozen"):
        run_holdout_records(drifted, events, freeze_record=record)


def test_holdout_metrics_ranges():
    entries = [
        {"label": "positive", "score": 0.0},
        {"label": "positive", "score": 0.0},
        {"label": "negative", "score": 0.0},
    ]
    metrics = compute_metrics(entries, [0.9, 0.8, 0.1], far=0.05)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["top1"] == 0.5  # one of two positives holds rank 1
    assert metrics["top5"] == 1.0
    assert metrics["burden_at_full_recall"] == 2.0
    for key in ("ap", "mrr", "top1", "top5", "recall_at_far"):
        assert 0.0 <= metrics[key] <= 1.0


def test_audit_development_flags_sealed():
    clean = F.audit_development([REPLAY])
    assert clean["ok"] and clean["sealed_sectors_touched"] == []
    dirty = F.audit_development([REPLAY, HOLDOUT])
    assert not dirty["ok"]
    assert dirty["sealed_sectors_touched"] == [82]
    assert HOLDOUT in dirty["offenders"]


def _live_checkpoint(tmp_path):
    """Pre-unblinding checkpoint: dev blind pairs, KELT-9 never seen."""
    import torch

    from tess_assoc.benchmark import build_benchmark
    from tess_assoc.extract import predicted_transits
    from tess_assoc.learn import prepare_split, train_model
    from tess_assoc.replay import load_replay_manifest, replay_blind_system

    replay = load_replay_manifest(REPLAY)
    systems = []
    for system in replay.systems:
        if system.tic_id == 16740101:
            continue
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
    parts = {s["tic_id"]: "train" for s in systems}
    bench = build_benchmark(systems, parts, dict(replay.matcher_thresholds))
    split = prepare_split(
        bench["positives"] + bench["negatives"], systems, parts,
        "morphology+scalars", ("train",), (),
    )
    assert split.train, "no dev training pairs"
    checkpoint = train_model(split.train, CONFIG, "morphology+scalars")
    ckpt_path = str(tmp_path / "holdout_ckpt.pt")
    torch.save(checkpoint["state_dict"], ckpt_path)
    return checkpoint, F.checkpoint_hash(checkpoint)


@needs_archive
def test_live_holdout_kelt9(tmp_path):
    from tess_assoc.holdout import render_holdout_report, run_holdout

    checkpoint, ckpt_sha = _live_checkpoint(tmp_path)
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        REPLAY, HOLDOUT, CONFIG, output_path=freeze_path,
        checkpoint_sha=ckpt_sha,
    )
    assert F.load_freeze_record(freeze_path).unblinded_utc is None
    manifest = F.load_holdout_manifest(HOLDOUT, freeze_path, CONFIG)
    results = run_holdout(
        manifest, freeze_path=freeze_path, checkpoint=checkpoint,
        ablation="morphology+scalars", config=CONFIG,
        cache_dir=str(tmp_path), log_path=str(tmp_path / "access.jsonl"),
    )
    assert results["sealed_sectors_touched"] == [82]
    assert results["n_test_positives"] >= 1
    assert results["verdict"]["verdict"] in ("GO", "PIVOT")
    assert results["verdict"]["reasons"]
    assert results["freeze"]["unblinded_utc"] is not None
    assert F.load_freeze_record(freeze_path).unblinded_utc is not None
    assert results["alias_reduction"]["median_reduction"] >= 0.0
    for method in ("deterministic", "learned"):
        for key in ("ap", "mrr", "top1", "recall_at_far"):
            assert 0.0 <= results[method][key] <= 1.0
    json.dumps(results)
    report = render_holdout_report(results)
    assert "Verdict" in report and "Sealed sectors touched: [82]" in report
    log_lines = (tmp_path / "access.jsonl").read_text().strip().split("\n")
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["event"] == "holdout_run"
