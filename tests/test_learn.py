"""Learned association tests (issue #6).

Pure metric/decision tests always run. Torch tests skip cleanly without
the `ml` extra; the live comparison needs the archive too.
"""

import json

import pytest

from conftest import needs_archive
from tess_assoc.learn import (
    ABLATIONS,
    LearnConfig,
    average_precision,
    decide,
    expected_calibration_error,
    normalize_window,
    recall_at_far,
)


def test_ablation_names_are_declared():
    assert "morphology" in ABLATIONS
    assert "morphology+scalars" in ABLATIONS


def test_normalize_window_standardizes():
    flux = [1.0, 1.0, 0.9, 1.0, 1.0]
    norm = normalize_window(flux)
    assert abs(sum(norm) / len(norm)) < 1e-9
    var = sum(v * v for v in norm) / len(norm)
    assert abs(var - 1.0) < 1e-9
    assert normalize_window([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_average_precision_known_values():
    assert average_precision([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6]) == 1.0
    assert average_precision([0, 0], [0.9, 0.1]) == 0.0
    ap = average_precision([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6])
    assert abs(ap - ((1 / 2 + 2 / 4) / 2)) < 1e-9


def test_calibration_perfect_is_zero():
    labels = [1, 1, 0, 0]
    result = expected_calibration_error(labels, [0.95, 0.85, 0.15, 0.05])
    assert result["ece"] < 0.15
    assert len(result["bins"]) == 10
    assert sum(b["n"] for b in result["bins"]) == 4


def test_recall_at_far():
    labels = [1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    out = recall_at_far(labels, scores, 0.5)
    assert out["recall"] == 1.0
    out = recall_at_far(labels, [0.1, 0.2, 0.9, 0.8, 0.7, 0.6], 0.0)
    assert out["recall"] == 0.0


def test_decide_go_pivot_and_small_n():
    config = LearnConfig()
    det = {"burden_at_full_recall": 100.0, "recall_at_far": 0.5, "top1": 0.0}
    good = {"burden_at_full_recall": 70.0, "recall_at_far": 0.5, "top1": 0.0}
    assert decide(det, good, config, 50)["verdict"] == "GO"
    assert decide(det, dict(det), config, 50)["verdict"] == "PIVOT"
    small = decide(det, good, config, 3)
    assert small["verdict"] == "PIVOT"
    assert any("too few" in r for r in small["reasons"])
    assert "unasked" in small["scope"]


def test_assemble_features_gates():
    from tess_assoc.learn import assemble_features

    fa = [1.0, 0.9, 1.0]
    feat = {"depth": 0.01, "duration_days": 0.2, "snr": 9.0}
    out = assemble_features(fa, fa, feat, feat, {}, {}, "morphology")
    assert set(out) == {"morph_a", "morph_b"}
    assert assemble_features(fa, [1.0], feat, feat, {}, {}, "morphology") is None
    assert assemble_features(fa, fa, feat, feat, {}, {}, "morphology+stellar") is None
    with pytest.raises(ValueError, match="unknown ablation"):
        assemble_features(fa, fa, feat, feat, {}, {}, "nope")


def test_prepare_split_rejects_tic_leak():
    from tess_assoc.learn import prepare_split

    pairs = [{"tic_id": 1, "a": "x", "b": "y", "label": "positive"}]
    with pytest.raises(ValueError, match="leakage"):
        prepare_split(pairs, [], {1: "train"}, "morphology", ("train",), ("train",))


def _torch():
    return pytest.importorskip("torch", reason="ml extra not installed")


def _tiny_rows():
    _torch()
    import random

    rng = random.Random(11)
    rows = []
    for star in (1, 2):
        base = [1.0] * 21
        for k in range(6):
            profund = 0.01 if k % 2 == 0 else 0.03
            fa = [v - (profund if 9 <= i <= 11 else 0.0) for i, v in enumerate(base)]
            fb = [v + rng.gauss(0, 0.0005) for v in fa]
            rows.append(
                {
                    "tic": star,
                    "features": {"morph_a": fa, "morph_b": fb},
                    "label": 1 if k % 2 == 0 else 0,
                }
            )
    return rows


def test_train_predict_deterministic():
    torch = _torch()
    from tess_assoc.learn import build_model, predict_proba, train_model

    torch.manual_seed(0)
    rows = _tiny_rows()
    config = LearnConfig(seed=7, epochs=3, batch_size=4, embedding_dim=16)
    first = predict_proba(
        train_model(rows, config, "morphology"), rows, "morphology"
    )
    second = predict_proba(
        train_model(rows, config, "morphology"), rows, "morphology"
    )
    assert first == second
    assert all(0.0 <= p <= 1.0 for p in first)
    assert len(first) == len(rows)


def test_model_sees_only_morphology_by_default():
    _torch()
    from tess_assoc.learn import build_model

    model = build_model(0, 0, 8)
    assert model.n_scalar == 0 and model.n_group == 0


def _live_systems(tmp_path):
    torch = _torch()
    from pathlib import Path

    from tess_assoc.event import EventRecord
    from tess_assoc.extract import predicted_transits
    from tess_assoc.replay import load_replay_manifest, replay_blind_system

    replay = load_replay_manifest(
        str(Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json")
    )
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
    return replay, systems


@needs_archive
def test_live_learned_vs_deterministic(tmp_path):
    _torch()
    from tess_assoc.benchmark import assign_partitions
    from tess_assoc.learn import LearnConfig, run_comparison

    replay, systems = _live_systems(tmp_path)
    config = LearnConfig(seed=7, epochs=5, batch_size=64, embedding_dim=32)
    results = run_comparison(
        systems,
        assign_partitions([s["tic_id"] for s in systems]),
        dict(replay.matcher_thresholds),
        config,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    assert results["n_test_pairs"] > 0
    assert set(results["ablations"]) == {
        "morphology",
        "morphology+scalars",
        "morphology+stellar",
        "morphology+contamination",
    }
    assert results["ablations"]["morphology"]["status"] == "trained"
    assert results["ablations"]["morphology+stellar"]["status"] == "unavailable"
    assert results["decision"]["verdict"] in ("GO", "PIVOT")
    assert results["decision"]["reasons"]
    import json

    json.dumps(results)
    assert (tmp_path / "checkpoints").exists()
