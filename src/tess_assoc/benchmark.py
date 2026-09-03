"""Development candidate-pair benchmark (issue #5).

Labels cross-sector recalled pairs as positives, same-TIC pairs as typed
hard negatives (timing-incompatible / morphology-matched / mismatch),
cross-TIC pairs as random negatives. Enforces TIC-level partitions,
ranks by the deterministic score, and reports retrieval, burden, and
alias reduction per slice — never mixing them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc._validate import is_finite_number, is_strict_int
from tess_assoc.event import EventRecord
from tess_assoc.manifest import ManifestSector, TracerManifest
from tess_assoc.matcher import MatchDecision, match, match_score
from tess_assoc.orbit import generate_aliases
from tess_assoc.replay import RECALL_TOL_DAYS
from tess_assoc.window import filter_aliases

PARTITIONS: tuple[str, ...] = ("train", "validation", "test")


def assign_partitions(tic_ids: list[int]) -> dict[int, str]:
    """Deterministic round-robin TIC assignment over train/validation/test."""
    if len(set(tic_ids)) != len(tic_ids):
        raise ValueError("duplicate TIC ids in partition assignment")
    assignments = {}
    for i, tic in enumerate(sorted(tic_ids)):
        assignments[tic] = PARTITIONS[i % len(PARTITIONS)]
    _protocol.validate_tic_partition(
        *[{t for t, p in assignments.items() if p == name} for name in PARTITIONS]
    )
    return assignments


def _recalled(
    records: dict[str, EventRecord], known: list[float], tol_days: float
) -> set[str]:
    return {
        rid
        for rid, rec in records.items()
        if any(abs(rec.t0 - t) <= tol_days for t in known)
    }


def _category(
    decision: MatchDecision, min_morph_corr: float, same_tic: bool
) -> str:
    if not same_tic:
        return "random"
    if not decision.timing_plausible:
        return "timing-incompatible"
    if decision.morph_corr >= min_morph_corr:
        return "morphology-matched"
    return "depth-duration-mismatch"


def _score_entry(
    tic_id: int | None, a_id: str, b_id: str, decision: MatchDecision
) -> dict[str, Any]:
    """Shared nine-key scoring record (one construction path)."""
    return {
        "tic_id": tic_id,
        "a": a_id,
        "b": b_id,
        "compatible": decision.compatible,
        "score": match_score(decision),
        "rel_depth_diff": decision.rel_depth_diff,
        "rel_duration_diff": decision.rel_duration_diff,
        "morph_corr": decision.morph_corr,
        "timing_plausible": decision.timing_plausible,
    }


def _validate_systems(systems: list[dict[str, Any]]) -> None:
    for system in systems:
        if not isinstance(system, dict):
            raise ValueError("each benchmark system must be a dict")
        for key in ("tic_id", "records", "known", "sectors"):
            if key not in system:
                raise ValueError(f"benchmark system missing key: {key}")
        if not isinstance(system["records"], dict) or not system["records"]:
            raise ValueError("system records must be a non-empty dict")
        if not all(
            isinstance(r, EventRecord) for r in system["records"].values()
        ):
            raise ValueError("system records must map ids to EventRecords")
        if not isinstance(system["known"], (list, tuple)):
            raise ValueError("system known times must be a list")
        if not all(is_finite_number(t) for t in system["known"]):
            raise ValueError("system known times must be finite numbers")
        if not isinstance(system.get("sectors", []), (list, tuple)):
            raise ValueError("system sectors must be a list")
        for entry in system.get("sectors", []):
            if (
                not isinstance(entry, dict)
                or "sector" not in entry
                or "windows" not in entry
            ):
                raise ValueError("each sector needs a sector id and windows")
            if not is_strict_int(entry["sector"]):
                raise ValueError("sector id must be an int")
            if not isinstance(entry["windows"], (list, tuple)):
                raise ValueError("sector windows must be a list")
    tics = [s["tic_id"] for s in systems]
    if len(set(tics)) != len(tics):
        raise ValueError("duplicate TIC ids in benchmark systems")


def rank_pairs(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank scored entries (shared deterministic/learned path).

    Entries need `score` and `label` ("positive"/other). Mutates ranks in.
    """
    ranked = sorted(entries, key=lambda p: p["score"], reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    pos_ranks = sorted(e["rank"] for e in ranked if e["label"] == "positive")
    mrr = sum(1.0 / r for r in pos_ranks) / len(pos_ranks) if pos_ranks else 0.0
    min_pos_score = min(
        (e["score"] for e in ranked if e["label"] == "positive"),
        default=float("-inf"),
    )
    return {
        "ranked_n": len(ranked),
        "positive_ranks": pos_ranks,
        "mrr": mrr,
        "burden_at_full_recall": sum(1 for e in ranked if e["score"] >= min_pos_score),
    }


def build_benchmark(
    systems: list[dict[str, Any]],
    partitions: dict[int, str],
    thresholds: dict[str, float],
    tol_days: float = RECALL_TOL_DAYS,
) -> dict[str, Any]:
    """Assemble labeled pairs, partitions, ranking, and alias accounting.

    Each system dict carries: tic_id, records {id: EventRecord}, known
    transit times, sectors [{sector, windows}], thresholds live in
    the shared `thresholds` arg. Positives are cross-sector recalled pairs
    (known repeated events — compatibility is measured, not assumed).
    """
    for tic in [s["tic_id"] for s in systems]:
        if tic not in partitions:
            raise ValueError(f"TIC {tic} has no partition assignment")
    for name in partitions.values():
        if name not in PARTITIONS:
            raise ValueError(f"unknown partition: {name}")
    _validate_systems(systems)
    _protocol.validate_tic_partition(
        *[
            {s["tic_id"] for s in systems if partitions[s["tic_id"]] == name}
            for name in PARTITIONS
        ]
    )

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for system in systems:
        tic = system["tic_id"]
        records = system["records"]
        recalled = _recalled(records, system["known"], tol_days)
        ids = sorted(records)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = records[ids[x]], records[ids[y]]
                decision = match(a, b, thresholds)
                entry = _score_entry(tic, ids[x], ids[y], decision)
                cross_sector = a.sector != b.sector
                if cross_sector and ids[x] in recalled and ids[y] in recalled:
                    entry["label"] = "positive"
                    positives.append(entry)
                else:
                    entry["label"] = "negative"
                    entry["category"] = _category(
                        decision, thresholds["min_morph_corr"], True
                    )
                    negatives.append(entry)
        for other in systems:
            if other["tic_id"] <= tic:
                continue
            for rid_a, rec_a in records.items():
                for rid_b, rec_b in other["records"].items():
                    decision = match(rec_a, rec_b, thresholds)
                    entry = _score_entry(
                        None, f"{tic}:{rid_a}", f"{other['tic_id']}:{rid_b}", decision
                    )
                    entry["label"] = "negative"
                    entry["category"] = _category(
                        decision, thresholds["min_morph_corr"], False
                    )
                    negatives.append(entry)

    ranking = rank_pairs(
        positives + [n for n in negatives if n["category"] != "random"]
    )

    alias_rows = []
    for entry in positives:
        owner = next(s for s in systems if s["tic_id"] == entry["tic_id"])
        recs = owner["records"]
        a, b = recs[entry["a"]], recs[entry["b"]]
        verdicts = _filter_with_windows(
            a, b, owner["sectors"], tol_days, thresholds, list(recs.values())
        )
        n_before = len(generate_aliases(abs(b.t0 - a.t0)))
        n_after = sum(1 for v in verdicts if v.retained)
        alias_rows.append(
            {
                "tic_id": entry["tic_id"],
                "a": entry["a"],
                "b": entry["b"],
                "aliases_before": n_before,
                "aliases_after": n_after,
            }
        )
    reductions = [
        (r["aliases_before"] - r["aliases_after"]) / r["aliases_before"]
        for r in alias_rows
        if r["aliases_before"]
    ]
    median_reduction = sorted(reductions)[len(reductions) // 2] if reductions else 0.0

    slices: dict[str, dict[str, Any]] = {}
    for name in ("random", "morphology-matched", "timing-incompatible",
                 "depth-duration-mismatch"):
        members = [n for n in negatives if n["category"] == name]
        slices[name] = {
            "n": len(members),
            "compatible_rate": (
                sum(1 for m in members if m["compatible"]) / len(members)
                if members
                else 0.0
            ),
            "max_score": max((m["score"] for m in members), default=float("-inf")),
        }

    return {
        "protocol_version": _protocol.PROTOCOL_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": dict(thresholds),
        "tol_days": tol_days,
        "partitions": dict(partitions),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "positives": positives,
        "negatives": negatives,
        "ranking": ranking,
        "alias_reduction": {
            "rows": alias_rows,
            "median_reduction": median_reduction,
        },
        "slices": slices,
    }


def _filter_with_windows(a, b, sector_windows, tol_days, thresholds, events):
    """Window filter over explicit observing windows (benchmark-owned)."""
    manifest = TracerManifest(
        name="benchmark-alias",
        tic_id=a.tic_id,
        epoch_match_tol_days=tol_days,
        matcher_thresholds=dict(thresholds),
        sectors=tuple(
            ManifestSector(sector=s["sector"], windows=tuple(s["windows"]))
            for s in sector_windows
        ),
        events=tuple(),
    )
    return filter_aliases(a, b, manifest, events)


def render_benchmark_report(results: dict[str, Any]) -> str:
    lines = [
        f"# Pair benchmark (protocol {results['protocol_version']})",
        f"{results['n_positives']} positives, {results['n_negatives']} negatives.",
        f"Partitions: {results['partitions']}",
        "",
        "## Ranking (positives vs same-TIC negatives)",
        f"MRR: {results['ranking']['mrr']:.3f}; "
        f"burden at full recall: {results['ranking']['burden_at_full_recall']}; "
        f"positive ranks: {results['ranking']['positive_ranks'][:10]}"
        f"{'...' if len(results['ranking']['positive_ranks']) > 10 else ''}",
        "",
        "## Slices",
    ]
    for name, s in results["slices"].items():
        if s["max_score"] != float("-inf"):
            lines.append(
                f"- {name}: n={s['n']}, compatible-rate={s['compatible_rate']:.3f}, "
                f"max-score={s['max_score']:.3f}"
            )
        else:
            lines.append(
                f"- {name}: n={s['n']}, compatible-rate={s['compatible_rate']:.3f} (empty)"
            )
    lines += [
        "",
        "## Alias reduction (positive pairs)",
        f"median reduction: {results['alias_reduction']['median_reduction']:.3f}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "PARTITIONS",
    "assign_partitions",
    "build_benchmark",
    "render_benchmark_report",
]
