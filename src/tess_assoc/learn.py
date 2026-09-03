"""Learned cross-epoch event association (issue #6).

Small Siamese 1D-CNN over normalized local transit morphology, with
predeclared ablations for scalar event features. TIC identifiers and
truth labels never enter the inputs. Evaluated on the identical pairs
and partitions as the deterministic baseline, then a written stop/go
decision records whether ML earns its place.

Requires the `ml` extra (torch) for training and prediction; the rest
imports cleanly without it.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

from tess_assoc.benchmark import build_benchmark, rank_pairs

SCALAR_KEYS: tuple[str, ...] = ("depth", "duration_days", "snr")
STELLAR_KEYS: tuple[str, ...] = ("r_star",)
CONTAMINATION_KEYS: tuple[str, ...] = ("crowding",)

ABLATIONS: tuple[str, ...] = (
    "morphology",
    "morphology+scalars",
    "morphology+stellar",
    "morphology+contamination",
)

PRIMARY_ABLATION = "morphology+scalars"


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "learned association needs the 'ml' extra: pip install tess-assoc[ml]"
        ) from e
    return torch


@dataclass(frozen=True)
class LearnConfig:
    seed: int = 7
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    embedding_dim: int = 64
    train_partitions: tuple[str, ...] = ("train", "validation")
    test_partitions: tuple[str, ...] = ("test",)
    min_burden_reduction: float = 0.20
    min_recall_gain: float = 0.05
    false_association_rate: float = 0.05


def normalize_window(flux: list[float]) -> list[float]:
    """Zero-mean, unit-variance morphology (no depth/duration leakage)."""
    if not flux:
        raise ValueError("flux window must be non-empty")
    mean = sum(flux) / len(flux)
    var = sum((v - mean) ** 2 for v in flux) / len(flux)
    std = math.sqrt(var) or 1e-9
    return [(v - mean) / std for v in flux]


def _scalar_vector(record: dict[str, float]) -> list[float]:
    """Raw log10 event features (v1 choice: no standardization)."""
    return [math.log10(max(record[k], 1e-12)) for k in SCALAR_KEYS]


def _group_vector(meta: dict[str, Any], keys: tuple[str, ...]) -> list[float] | None:
    if any(k not in meta for k in keys):
        return None
    try:
        return [float(meta[k]) for k in keys]
    except (TypeError, ValueError):
        return None


def assemble_features(
    fa: list[float],
    fb: list[float],
    feat_a: dict[str, float],
    feat_b: dict[str, float],
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
    ablation: str,
) -> dict[str, Any] | None:
    """Feature dict for one labeled pair, or None if the ablation lacks data.

    Only morphology plus the ablated scalar groups — TIC ids and labels
    never enter the features.
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    if len(fa) != len(fb) or not fa:
        return None
    out: dict[str, Any] = {
        "morph_a": normalize_window(fa),
        "morph_b": normalize_window(fb),
    }
    if ablation == "morphology":
        return out
    scalars_a = _scalar_vector(feat_a)
    scalars_b = _scalar_vector(feat_b)
    if ablation == "morphology+scalars":
        out["scalars_a"] = scalars_a
        out["scalars_b"] = scalars_b
        return out
    meta_keys = STELLAR_KEYS if ablation == "morphology+stellar" else CONTAMINATION_KEYS
    group_a = _group_vector(meta_a, meta_keys)
    group_b = _group_vector(meta_b, meta_keys)
    if group_a is None or group_b is None:
        return None
    out["scalars_a"] = scalars_a
    out["scalars_b"] = scalars_b
    out["group_a"] = group_a
    out["group_b"] = group_b
    return out


def build_model(n_scalar: int, n_group: int, embedding_dim: int = 64):
    """Small shared-encoder Siamese network (torch required)."""
    torch = _require_torch()
    import torch.nn as nn

    class SiameseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
                nn.Linear(32 * 4, embedding_dim), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(embedding_dim + n_scalar + n_group, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.n_scalar = n_scalar
            self.n_group = n_group

        def forward(self, ma, mb, sa=None, sb=None, ga=None, gb=None):
            ea = self.encoder(ma.unsqueeze(1))
            eb = self.encoder(mb.unsqueeze(1))
            parts = [torch.abs(ea - eb)]
            if self.n_scalar:
                parts += [torch.abs(sa - sb)]
            if self.n_group:
                parts += [torch.abs(ga - gb)]
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    return SiameseNet()


@dataclass
class SplitData:
    train: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)
    skipped_length: int = 0
    skipped_ablation: int = 0


def _record_features(rec) -> tuple[dict[str, float], dict[str, float], list[float]]:
    """Scalar features, stellar metadata, and raw flux from one record."""
    return (
        {"depth": rec.depth, "duration_days": rec.duration_days, "snr": rec.snr},
        dict(rec.stellar_meta),
        list(rec.local_flux),
    )


def prepare_split(
    pairs: list[dict[str, Any]],
    systems: list[dict[str, Any]],
    partitions: dict[int, str],
    ablation: str,
    train_partitions: tuple[str, ...],
    test_partitions: tuple[str, ...],
) -> SplitData:
    """Assemble labeled feature pairs with strict TIC-level splitting.

    A TIC in train can never appear in test (star-level leakage guard).
    Only same-TIC pairs (with a tic_id) are eligible; random cross-star
    pairs are reported separately, never trained or ranked here.
    """
    train_tics = {t for t, p in partitions.items() if p in train_partitions}
    test_tics = {t for t, p in partitions.items() if p in test_partitions}
    if train_tics & test_tics:
        raise ValueError(f"TIC leakage across splits: {train_tics & test_tics}")
    train_tics = {t for t, p in partitions.items() if p in train_partitions}
    test_tics = {t for t, p in partitions.items() if p in test_partitions}
    if train_tics & test_tics:
        raise ValueError(f"TIC leakage across splits: {train_tics & test_tics}")
    by_tic = {s["tic_id"]: s for s in systems}
    split = SplitData()
    for pair in pairs:
        tic = pair["tic_id"]
        if tic is None:
            continue
        part = partitions.get(tic)
        if part is None:
            raise ValueError(f"TIC {tic} has no partition assignment")
        if part not in train_partitions and part not in test_partitions:
            continue
        recs = by_tic[tic]["records"]
        feat_a, meta_a, fa = _record_features(recs[pair["a"]])
        feat_b, meta_b, fb = _record_features(recs[pair["b"]])
        if len(fa) != len(fb) or not fa:
            split.skipped_length += 1
            continue
        feats = assemble_features(fa, fb, feat_a, feat_b, meta_a, meta_b, ablation)
        if feats is None:
            split.skipped_ablation += 1
            continue
        row = {"features": feats, "label": 1 if pair["label"] == "positive" else 0}
        if part in train_partitions:
            split.train.append(row)
        else:
            split.test.append(row)
    return split


def _batch_tensors(rows: list[dict[str, Any]], idx: list[int], keys: tuple[str, ...]):
    torch = _require_torch()
    import torch as _t

    out = {}
    for key in keys:
        if key in rows[idx[0]]["features"]:
            out[key] = _t.tensor(
                [rows[i]["features"][key] for i in idx], dtype=_t.float32
            )
    labels = _t.tensor([rows[i]["label"] for i in idx], dtype=_t.float32)
    return out, labels


def _feature_keys(ablation: str) -> tuple[tuple[str, ...], int, int]:
    if ablation == "morphology":
        return (("morph_a", "morph_b"), 0, 0)
    if ablation == "morphology+scalars":
        return (("morph_a", "morph_b", "scalars_a", "scalars_b"), len(SCALAR_KEYS), 0)
    group_len = len(STELLAR_KEYS if ablation == "morphology+stellar" else CONTAMINATION_KEYS)
    return (
        ("morph_a", "morph_b", "scalars_a", "scalars_b", "group_a", "group_b"),
        len(SCALAR_KEYS),
        group_len,
    )


def train_model(
    train_rows: list[dict[str, Any]], config: LearnConfig, ablation: str
) -> dict[str, Any]:
    """Train one ablation; return weights state plus training provenance."""
    torch = _require_torch()
    import random as _random
    import numpy as _np

    _random.seed(config.seed)
    _np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    keys, n_scalar, n_group = _feature_keys(ablation)
    model = build_model(n_scalar, n_group, config.embedding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    order = list(range(len(train_rows)))
    for _ in range(config.epochs):
        _random.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            feats, labels = _batch_tensors(train_rows, idx, keys)
            opt.zero_grad()
            logits = model(
                feats["morph_a"], feats["morph_b"],
                feats.get("scalars_a"), feats.get("scalars_b"),
                feats.get("group_a"), feats.get("group_b"),
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_scalar": n_scalar,
        "n_group": n_group,
        "embedding_dim": config.embedding_dim,
        "n_params": n_params,
        "n_train": len(train_rows),
        "seed": config.seed,
        "epochs": config.epochs,
    }


def predict_proba(
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    ablation: str,
) -> list[float]:
    """P(same transit-producing object) for each row (no gradients)."""
    torch = _require_torch()
    keys, n_scalar, n_group = _feature_keys(ablation)
    model = build_model(n_scalar, n_group, checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            chunk = rows[i : i + 64]
            feats, _ = _batch_tensors(chunk, list(range(len(chunk))), keys)
            logits = model(
                feats["morph_a"], feats["morph_b"],
                feats.get("scalars_a"), feats.get("scalars_b"),
                feats.get("group_a"), feats.get("group_b"),
            )
            out.extend(float(v) for v in torch.sigmoid(logits).tolist())
    return out


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Area under the precision-recall curve via ranking integration."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    total_pos = sum(labels)
    if not total_pos:
        return 0.0
    hits, ap = 0, 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            hits += 1
            ap += hits / rank
    return ap / total_pos


def expected_calibration_error(
    labels: list[int], probas: list[float], n_bins: int = 10
) -> dict[str, Any]:
    """ECE plus per-bin reliability for probability outputs."""
    bins: list[dict[str, Any]] = []
    ece, n = 0.0, len(probas)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probas) if (lo < p <= hi) or (b == 0 and p == 0.0)]
        if not idx:
            bins.append({"bin": [lo, hi], "n": 0, "mean_proba": 0.0, "accuracy": 0.0})
            continue
        mean_p = sum(probas[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += abs(mean_p - acc) * len(idx) / n
        bins.append({"bin": [lo, hi], "n": len(idx), "mean_proba": mean_p, "accuracy": acc})
    return {"ece": ece, "bins": bins}


def recall_at_far(
    labels: list[int], scores: list[float], far: float
) -> dict[str, float]:
    """Recall at a fixed false-association rate (threshold from negatives)."""
    negs = sorted((s for l, s in zip(labels, scores) if not l), reverse=True)
    if not negs:
        return {"threshold": 0.0, "recall": 0.0, "far": far}
    cut = max(1, math.ceil(far * len(negs)))
    threshold = negs[cut - 1] if cut <= len(negs) else float("-inf")
    pos = [s for l, s in zip(labels, scores) if l]
    recall = sum(1 for s in pos if s >= threshold) / len(pos) if pos else 0.0
    return {"threshold": threshold, "recall": recall, "far": far}


def decide(
    det_metrics: dict[str, float],
    learned_metrics: dict[str, float],
    config: LearnConfig,
    n_test_positives: int,
) -> dict[str, Any]:
    """Written stop/go decision with predeclared materiality margins.

    Scope honesty: the verdict below covers hot-Jupiter data, where
    same-star transits are morphologically identical and timing alone
    separates pairs. On such data a shape-only model cannot win by
    construction, so a PIVOT here means "unasked on informative data",
    not "machine learning fails". No final GO/PIVOT claim on machine
    learning itself until a rematch on hard negatives with
    shape-discriminable signal (injections, issue #7).
    """
    burden_gain = (
        (det_metrics["burden_at_full_recall"] - learned_metrics["burden_at_full_recall"])
        / det_metrics["burden_at_full_recall"]
        if det_metrics["burden_at_full_recall"]
        else 0.0
    )
    recall_gain = learned_metrics["recall_at_far"] - det_metrics["recall_at_far"]
    top1_gain = learned_metrics["top1"] - det_metrics["top1"]
    reasons = [
        f"burden change {burden_gain:+.2%} (needs {config.min_burden_reduction:.0%})",
        f"recall@FAR change {recall_gain:+.3f} (needs {config.min_recall_gain:.2f})",
        f"top-1 change {top1_gain:+.3f}",
        f"test positives: {n_test_positives}",
    ]
    go = (
        burden_gain >= config.min_burden_reduction
        or recall_gain >= config.min_recall_gain
    )
    scope = (
        "hot-Jupiter data only: same-star transits are morphologically "
        "identical, so timing alone separates pairs and a shape-only model "
        "cannot win by construction. This verdict means the ML question is "
        "unasked on informative data, not answered; no final claim until a "
        "rematch on hard negatives with shape-discriminable signal (#7)."
    )
    if n_test_positives < 10:
        return {
            "verdict": "PIVOT",
            "scope": scope,
            "reasons": reasons + ["too few test positives to judge ML"],
        }
    if go:
        return {"verdict": "GO", "scope": scope, "reasons": reasons}
    return {
        "verdict": "PIVOT",
        "scope": scope,
        "reasons": reasons + ["deterministic baseline remains the supported path"],
    }


def _test_metrics(
    entries: list[dict[str, Any]], scores: list[float], far: float
) -> dict[str, float]:
    """Head-to-head metrics on identical pairs.

    top1/top5 are the fraction of test positives ranked at position <= k —
    harsh with hundreds of positives, but identical for both methods.
    """
    labels = [1 if e["label"] == "positive" else 0 for e in entries]
    rescored = [dict(e, score=s) for e, s in zip(entries, scores)]
    ranking = rank_pairs(rescored)
    ranks = ranking["positive_ranks"]
    return {
        "ap": average_precision(labels, scores),
        "mrr": ranking["mrr"],
        "burden_at_full_recall": float(ranking["burden_at_full_recall"]),
        "top1": sum(1 for r in ranks if r <= 1) / len(ranks) if ranks else 0.0,
        "top5": sum(1 for r in ranks if r <= 5) / len(ranks) if ranks else 0.0,
        "recall_at_far": recall_at_far(labels, scores, far)["recall"],
    }


def run_comparison(
    systems: list[dict[str, Any]],
    partitions: dict[int, str],
    thresholds: dict[str, float],
    config: LearnConfig,
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    """Learned vs deterministic on identical pairs and partitions + decision."""
    bench = build_benchmark(systems, partitions, thresholds)
    test_tics = {t for t, p in partitions.items() if p in config.test_partitions}
    test_entries = [
        e
        for e in bench["positives"] + bench["negatives"]
        if e.get("tic_id") in test_tics
    ]
    det_scores = [e["score"] for e in test_entries]
    det_metrics = _test_metrics(test_entries, det_scores, config.false_association_rate)

    ablation_results: dict[str, Any] = {}
    learned_primary: dict[str, float] | None = None
    for ablation in ABLATIONS:
        split = prepare_split(
            bench["positives"] + bench["negatives"],
            systems,
            partitions,
            ablation,
            config.train_partitions,
            config.test_partitions,
        )
        if not split.train and not split.test and split.skipped_ablation:
            ablation_results[ablation] = {
                "status": "unavailable",
                "reason": "required features absent from records",
                "n_train": 0,
                "n_test": 0,
            }
            continue
        if not split.train or not split.test:
            ablation_results[ablation] = {
                "status": "unavailable",
                "reason": "empty train or test split",
                "n_train": len(split.train),
                "n_test": len(split.test),
            }
            continue
        checkpoint = train_model(split.train, config, ablation)
        probas = predict_proba(checkpoint, split.test, ablation)
        test_labels = [r["label"] for r in split.test]
        metrics = _test_metrics(
            [
                {"label": ("positive" if r["label"] else "negative")}
                for r in split.test
            ],
            probas,
            config.false_association_rate,
        )
        metrics["calibration"] = expected_calibration_error(test_labels, probas)
        saved_at = None
        if checkpoint_dir is not None:
            torch = _require_torch()
            os.makedirs(checkpoint_dir, exist_ok=True)
            saved_at = os.path.join(checkpoint_dir, f"siamese_{ablation.replace('+', '_')}.pt")
            torch.save(checkpoint["state_dict"], saved_at)
        ablation_results[ablation] = {
            "status": "trained",
            "n_train": len(split.train),
            "n_test": len(split.test),
            "skipped_length": split.skipped_length,
            "skipped_ablation": split.skipped_ablation,
            "metrics": metrics,
            "checkpoint": saved_at,
            "seed": config.seed,
        }
        if ablation == PRIMARY_ABLATION:
            learned_primary = metrics
    if learned_primary is None:
        learned_primary = ablation_results["morphology"].get("metrics", det_metrics)
    decision = decide(
        det_metrics,
        learned_primary,
        config,
        sum(1 for e in test_entries if e["label"] == "positive"),
    )
    return {
        "protocol_version": bench["protocol_version"],
        "config": {
            "seed": config.seed,
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "embedding_dim": config.embedding_dim,
            "train_partitions": list(config.train_partitions),
            "test_partitions": list(config.test_partitions),
            "min_burden_reduction": config.min_burden_reduction,
            "min_recall_gain": config.min_recall_gain,
            "false_association_rate": config.false_association_rate,
        },
        "partitions": dict(partitions),
        "n_test_pairs": len(test_entries),
        "deterministic": det_metrics,
        "ablations": ablation_results,
        "decision": decision,
    }


__all__ = [
    "ABLATIONS",
    "LearnConfig",
    "average_precision",
    "build_model",
    "decide",
    "expected_calibration_error",
    "normalize_window",
    "predict_proba",
    "prepare_split",
    "recall_at_far",
    "run_comparison",
    "train_model",
]
