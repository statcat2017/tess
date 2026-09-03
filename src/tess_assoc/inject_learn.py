"""Leave-one-TIC-out learned scoring over injection cells (issue #7).

Trains on two TICs' full constructed candidate sets, scores the third —
same harness as issue #6. Reports deterministic correctness and learned
correctness by operating region (depth x shape-pair x separation), beside
aggregate ranking metrics and per-experiment candidate burden.
"""

from __future__ import annotations

from typing import Any

from tess_assoc.benchmark import rank_pairs


def _region_separation(cell: dict[str, Any]) -> str:
    """One definition of separation: the stored cell value, always."""
    separation = cell.get("separation")
    if separation in ("same-epoch", "cross-epoch"):
        return separation
    return "same-epoch" if cell["sectors"][0] == cell["sectors"][1] else "cross-epoch"


def learned_comparison(
    rows: list[dict[str, Any]], config, ablation: str = "morphology+scalars"
) -> dict[str, Any]:
    """Leave-one-TIC-out learned scoring over study pairs."""
    from tess_assoc.learn import (
        assemble_features,
        average_precision,
        predict_proba,
        train_model,
    )

    labeled = [
        (cell, candidate)
        for cell in rows
        for candidate in cell.get("learn_candidates", [])
    ]
    tics = sorted({c["tic_id"] for c in rows})
    rotations = []
    for test_tic in tics:
        train_rows, test_rows = [], []
        for cell, learned in labeled:
            feats = assemble_features(
                learned["fa"], learned["fb"],
                learned["feat_a"], learned["feat_b"], {}, {},
                ablation,
            )
            if feats is None:
                continue
            row = {
                "features": feats,
                "label": learned["label"],
                "cell": cell,
                "candidate": learned,
            }
            (test_rows if cell["tic_id"] == test_tic else train_rows).append(row)
        if not train_rows or not test_rows:
            continue
        checkpoint = train_model(train_rows, config, ablation)
        probas = predict_proba(checkpoint, test_rows, ablation)
        labels = [r["label"] for r in test_rows]
        learned_ranking = rank_pairs(
            [
                {"score": score, "label": "positive" if row["label"] else "negative"}
                for row, score in zip(test_rows, probas)
            ]
        )
        candidate_groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for row, score in zip(test_rows, probas):
            candidate_groups.setdefault(row["candidate"]["group"], []).append(
                (row["candidate"], score)
            )
        candidate_rankings = []
        for group, candidates in sorted(candidate_groups.items()):
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            positive_scores = [score for candidate, score in ranked if candidate["target"]]
            if not positive_scores:
                continue
            target_score = min(positive_scores)
            target_rank = min(
                rank for rank, (candidate, _) in enumerate(ranked, start=1)
                if candidate["target"]
            )
            candidate_rankings.append(
                {
                    "group": group,
                    "n_candidates": len(ranked),
                    "target_rank": target_rank,
                    "burden_at_target": sum(score >= target_score for _, score in ranked),
                }
            )
        regions: dict[
            tuple[float, str, str, str],
            list[tuple[dict[str, Any], dict[str, Any], float]],
        ] = {}
        for scored, proba in zip(test_rows, probas):
            cell = scored["cell"]
            key = (
                cell["depth"], cell["shape"],
                cell.get("shape_b", cell["shape"]), _region_separation(cell),
            )
            regions.setdefault(key, []).append((cell, scored["candidate"], proba))
        totals: dict[tuple[float, str, str, str], int] = {}
        for cell in rows:
            if cell["tic_id"] != test_tic:
                continue
            key = (
                cell["depth"], cell["shape"],
                cell.get("shape_b", cell["shape"]), _region_separation(cell),
            )
            totals[key] = totals.get(key, 0) + 1
        region_metrics = []
        for key in sorted(totals):
            depth, shape, shape_b, separation = key
            members = regions.get(key, [])
            positives = [p for _, candidate, p in members if candidate["label"]]
            negatives = [p for _, candidate, p in members if not candidate["label"]]
            region_metrics.append(
                {
                    "depth": depth,
                    "shape": shape,
                    "shape_b": shape_b,
                    "separation": separation,
                    "n_cells": totals[key],
                    "n_scored_pairs": len(members),
                    "pairs_per_cell": len(members) / totals[key],
                    "n_positive_pairs": len(positives),
                    "n_negative_pairs": len(negatives),
                    "learned_mean_score": (
                        sum(p for _, _, p in members) / len(members) if members else None
                    ),
                    "learned_mean_positive_score": (
                        sum(positives) / len(positives) if positives else None
                    ),
                    "learned_mean_negative_score": (
                        sum(negatives) / len(negatives) if negatives else None
                    ),
                    "learned_correct_rate": (
                        sum(
                            (p >= 0.5) == bool(candidate["label"])
                            for cell, candidate, p in members
                        ) / len(members)
                        if members else None
                    ),
                    "deterministic_correct_rate": (
                        sum(
                            cell["compatible"] == bool(candidate["label"])
                            for cell, candidate, _ in members
                        ) / len(members)
                        if members else None
                    ),
                }
            )
        rotations.append(
            {
                "test_tic": test_tic,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "checkpoint": {
                    "n_params": checkpoint["n_params"],
                    "seed": checkpoint["seed"],
                    "epochs": checkpoint["epochs"],
                },
                "ap": average_precision(labels, probas),
                "mrr": learned_ranking["mrr"],
                "burden_at_full_recall": learned_ranking["burden_at_full_recall"],
                "candidate_rankings": candidate_rankings,
                "mean_proba_positive": sum(
                    p for p, r in zip(probas, test_rows) if r["label"]
                ) / max(1, sum(r["label"] for r in test_rows)),
                "operating_regions": region_metrics,
            }
        )
    return {
        "ablation": ablation,
        "config": {
            "seed": config.seed,
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "embedding_dim": config.embedding_dim,
        },
        "rotations": rotations,
    }


__all__ = ["learned_comparison"]
