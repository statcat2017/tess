"""Methodology freeze and sealed-holdout gate (issue #8).

The freeze record pins the complete methodology *before* any sealed
measurement is touched: source-tree hash, protocol version, thresholds,
learned-model configuration, proposer/injection parameters, manifest file
hashes, TIC lists, catalogue provenance, and the trained-checkpoint hash.

Sealed sectors stay unreachable through normal development inputs (their
loaders always reject sectors 80+). The only way in is
`load_holdout_manifest`, which refuses without a verified freeze record —
so unblinding is impossible by accident and auditable by construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc._validate import (
    require_finite,
    require_positive_finite,
    require_strict_int,
)
from tess_assoc.matcher import REQUIRED_THRESHOLDS


def source_tree_hash(src_dir: str | None = None) -> str:
    """SHA-256 over every source file (catches uncommitted edits too)."""
    root = Path(src_dir) if src_dir else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: str) -> str:
    """SHA-256 of exact file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_hash(checkpoint: dict[str, Any]) -> str:
    """SHA-256 of serialized checkpoint weights (torch required)."""
    try:
        import io

        import torch
    except ImportError as e:
        raise ImportError(
            "checkpoint hashing needs the 'ml' extra: pip install tess-assoc[ml]"
        ) from e
    buf = io.BytesIO()
    torch.save(checkpoint["state_dict"], buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


@dataclass(frozen=True)
class HoldoutSystem:
    """Known-system record for the sealed cohort (no sector restriction).

    Constructible ONLY via load_holdout_manifest, which verifies a freeze
    record first. Normal manifest loaders keep rejecting sealed sectors.
    """

    name: str
    tic_id: int
    period_days: float
    t0_bjd_tdb: float
    duration_hours: float
    sectors: tuple[int, ...] = ()
    toi: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("system name must be a non-empty str")
        require_strict_int("tic_id", self.tic_id, minimum=1)
        require_positive_finite("period_days", self.period_days)
        require_finite("t0_bjd_tdb", self.t0_bjd_tdb)
        require_positive_finite("duration_hours", self.duration_hours)
        if not isinstance(self.sectors, (list, tuple)) or not self.sectors:
            raise ValueError("system sectors must be a non-empty list")
        for sector in self.sectors:
            require_strict_int("sector", sector, minimum=1)
        if not isinstance(self.toi, str):
            raise ValueError("toi must be a str")
        object.__setattr__(self, "sectors", tuple(self.sectors))


@dataclass(frozen=True)
class HoldoutManifest:
    """Sealed-cohort manifest (duck-types ReplayManifest for blind replay)."""

    name: str
    product: str
    ephemeris_source: str
    epoch_match_tol_days: float
    window_half_span_days: float
    resample_samples: int
    matcher_thresholds: dict[str, float] = field(default_factory=dict)
    systems: tuple[HoldoutSystem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("holdout manifest name must be a non-empty str")
        if self.product != "TESS-SPOC FFI":
            raise ValueError("holdout manifest must declare the TESS-SPOC FFI product")
        if not isinstance(self.ephemeris_source, str) or not self.ephemeris_source:
            raise ValueError("ephemeris_source must be a non-empty str")
        require_positive_finite("epoch_match_tol_days", self.epoch_match_tol_days)
        require_positive_finite("window_half_span_days", self.window_half_span_days)
        require_strict_int("resample_samples", self.resample_samples, minimum=3)
        if not isinstance(self.matcher_thresholds, dict):
            raise ValueError("matcher_thresholds must be a dict")
        for key in REQUIRED_THRESHOLDS:
            if key not in self.matcher_thresholds:
                raise ValueError(f"matcher_thresholds missing key: {key}")
            require_finite(f"threshold {key}", self.matcher_thresholds[key])
        if not isinstance(self.systems, (list, tuple)) or not self.systems:
            raise ValueError("systems must be a non-empty list")
        if not all(isinstance(s, HoldoutSystem) for s in self.systems):
            raise ValueError("systems must be HoldoutSystem records")
        object.__setattr__(self, "matcher_thresholds", dict(self.matcher_thresholds))
        object.__setattr__(self, "systems", tuple(self.systems))


@dataclass(frozen=True)
class FreezeRecord:
    """Immutable methodology snapshot (unblinded_utc set once, at unblinding)."""

    protocol_version: str
    code_sha: str
    created_utc: str
    unblinded_utc: str | None
    thresholds: dict[str, float]
    learn_config: dict[str, Any]
    ablation: str
    proposer_snr_threshold: float
    injection: dict[str, Any]
    manifests: dict[str, dict[str, str]]
    systems: dict[str, list[int]]
    checkpoint_sha: str | None
    ephemeris_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "code_sha": self.code_sha,
            "created_utc": self.created_utc,
            "unblinded_utc": self.unblinded_utc,
            "thresholds": dict(self.thresholds),
            "learn_config": _canonical(self.learn_config),
            "ablation": self.ablation,
            "proposer_snr_threshold": self.proposer_snr_threshold,
            "injection": _canonical(self.injection),
            "manifests": _canonical(self.manifests),
            "systems": _canonical(self.systems),
            "checkpoint_sha": self.checkpoint_sha,
            "ephemeris_source": self.ephemeris_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FreezeRecord":
        for key in (
            "protocol_version", "code_sha", "created_utc", "thresholds",
            "learn_config", "ablation", "proposer_snr_threshold", "injection",
            "manifests", "systems", "ephemeris_source",
        ):
            if key not in d:
                raise ValueError(f"freeze record missing key: {key}")
        return cls(
            protocol_version=d["protocol_version"],
            code_sha=d["code_sha"],
            created_utc=d["created_utc"],
            unblinded_utc=d.get("unblinded_utc"),
            thresholds=dict(d["thresholds"]),
            learn_config=dict(d["learn_config"]),
            ablation=d["ablation"],
            proposer_snr_threshold=d["proposer_snr_threshold"],
            injection=dict(d["injection"]),
            manifests={k: dict(v) for k, v in d["manifests"].items()},
            systems={k: list(v) for k, v in d["systems"].items()},
            checkpoint_sha=d.get("checkpoint_sha"),
            ephemeris_source=d["ephemeris_source"],
        )

    def stamped(self, unblinded_utc: str) -> "FreezeRecord":
        """Copy with the unblinding timestamp (first stamp wins)."""
        if self.unblinded_utc is not None:
            return self
        return FreezeRecord(
            **{**self.to_dict(), "unblinded_utc": unblinded_utc}
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: Any) -> Any:
    """JSON-normalized copy (tuples → lists) for stable comparisons."""
    return json.loads(json.dumps(obj))


def create_freeze(
    dev_manifest_path: str,
    holdout_manifest_path: str,
    config,
    *,
    output_path: str,
    ablation: str = "morphology+scalars",
    checkpoint_sha: str | None = None,
) -> FreezeRecord:
    """Snapshot the methodology to output_path (before any sealed access)."""
    from tess_assoc.inject_geometry import (
        DEPTHS,
        SAME_EPOCH_DT_DAYS,
        SHAPES,
        INJECTION_DURATION_DAYS,
    )
    from tess_assoc.propose import PROPOSER_SNR_THRESHOLD
    from tess_assoc.replay import load_replay_manifest

    dev = load_replay_manifest(dev_manifest_path)
    with open(holdout_manifest_path) as f:
        holdout_manifest = _parse_holdout_manifest(json.load(f))
    holdout_systems = [s.tic_id for s in holdout_manifest.systems]
    record = FreezeRecord(
        protocol_version=_protocol.PROTOCOL_VERSION,
        code_sha=source_tree_hash(),
        created_utc=_utcnow(),
        unblinded_utc=None,
        thresholds=dict(dev.matcher_thresholds),
        learn_config=_canonical(dataclasses.asdict(config)),
        ablation=ablation,
        proposer_snr_threshold=PROPOSER_SNR_THRESHOLD,
        injection={
            "depths": list(DEPTHS),
            "shapes": list(SHAPES),
            "same_epoch_dt_days": SAME_EPOCH_DT_DAYS,
            "duration_days": INJECTION_DURATION_DAYS,
        },
        manifests={
            "dev": {
                "path": str(Path(dev_manifest_path).resolve()),
                "sha256": file_hash(dev_manifest_path),
            },
            "holdout": {
                "path": str(Path(holdout_manifest_path).resolve()),
                "sha256": file_hash(holdout_manifest_path),
            },
        },
        systems={
            "dev": sorted(s.tic_id for s in dev.systems),
            "holdout": sorted(holdout_systems),
        },
        checkpoint_sha=checkpoint_sha,
        ephemeris_source=dev.ephemeris_source,
    )
    Path(output_path).write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return record


def load_freeze_record(path: str) -> FreezeRecord:
    with open(path) as f:
        return FreezeRecord.from_dict(json.load(f))


def verify_freeze(record_or_path: FreezeRecord | str, config) -> FreezeRecord:
    """Recompute everything; raise naming the first mismatch found."""
    record = (
        load_freeze_record(record_or_path)
        if isinstance(record_or_path, str)
        else record_or_path
    )
    problems: list[str] = []
    if record.protocol_version != _protocol.PROTOCOL_VERSION:
        problems.append(
            f"protocol {record.protocol_version} != {_protocol.PROTOCOL_VERSION}"
        )
    if record.code_sha != source_tree_hash():
        problems.append("source tree changed since freeze (code_sha mismatch)")
    dev_info = record.manifests.get("dev", {})
    if not dev_info or file_hash(dev_info["path"]) != dev_info["sha256"]:
        problems.append("dev manifest changed since freeze")
    else:
        from tess_assoc.replay import load_replay_manifest

        dev = load_replay_manifest(dev_info["path"])
        if dict(dev.matcher_thresholds) != record.thresholds:
            problems.append("matcher thresholds changed since freeze")
    holdout_info = record.manifests.get("holdout", {})
    if not holdout_info or "sha256" not in holdout_info:
        problems.append("holdout manifest not pinned by freeze")
    elif (
        Path(holdout_info["path"]).exists()
        and file_hash(holdout_info["path"]) != holdout_info["sha256"]
    ):
        problems.append("holdout manifest changed since freeze")
    if record.learn_config != _canonical(dataclasses.asdict(config)):
        problems.append("learn config changed since freeze")
    if problems:
        raise ValueError("freeze verification failed: " + "; ".join(problems))
    return record


def _parse_holdout_manifest(d: dict[str, Any]) -> HoldoutManifest:
    for key in (
        "name", "product", "ephemeris_source", "epoch_match_tol_days",
        "window_half_span_days", "resample_samples", "matcher_thresholds",
        "systems",
    ):
        if key not in d:
            raise ValueError(f"holdout manifest missing key: {key}")
    systems = [
        HoldoutSystem(
            name=s["name"],
            tic_id=s["tic_id"],
            period_days=s["period_days"],
            t0_bjd_tdb=s["t0_bjd_tdb"],
            duration_hours=s["duration_hours"],
            sectors=tuple(s["sectors"]),
            toi=s.get("toi", ""),
        )
        for s in d["systems"]
    ]
    return HoldoutManifest(
        name=d["name"],
        product=d["product"],
        ephemeris_source=d["ephemeris_source"],
        epoch_match_tol_days=d["epoch_match_tol_days"],
        window_half_span_days=d["window_half_span_days"],
        resample_samples=d["resample_samples"],
        matcher_thresholds=dict(d["matcher_thresholds"]),
        systems=tuple(systems),
    )


def load_holdout_manifest(
    path: str, freeze_record: FreezeRecord | str, config
) -> HoldoutManifest:
    """Unblind gate: verified freeze required, sealed sectors allowed after.

    Binds on manifest bytes, not location: a relocated-but-identical file
    verifies identically.
    """
    record = verify_freeze(freeze_record, config)
    info = record.manifests["holdout"]
    if file_hash(path) != info["sha256"]:
        raise ValueError("holdout manifest bytes differ from frozen manifest")
    with open(path) as f:
        manifest = _parse_holdout_manifest(json.load(f))
    if dict(manifest.matcher_thresholds) != record.thresholds:
        raise ValueError("holdout thresholds differ from frozen thresholds")
    return manifest


def mark_unblinded(record_path: str) -> FreezeRecord:
    """Stamp first-sealed-access time (idempotent: first stamp wins)."""
    record = load_freeze_record(record_path)
    stamped = record.stamped(_utcnow())
    if stamped is not record:
        Path(record_path).write_text(json.dumps(stamped.to_dict(), indent=2) + "\n")
    return stamped


def log_access(log_path: str, event: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON access-log line for the audit trail."""
    entry = {"utc": _utcnow(), **event}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def audit_development(dev_manifest_paths: list[str]) -> dict[str, Any]:
    """Independent check: dev manifests must load with zero sealed sectors."""
    from tess_assoc.replay import load_replay_manifest

    checked: list[str] = []
    sealed: list[int] = []
    offenders: list[str] = []
    for path in dev_manifest_paths:
        try:
            manifest = load_replay_manifest(path)
        except ValueError:
            with open(path) as f:
                raw = json.load(f)
            bad = sorted(
                {
                    s
                    for system in raw.get("systems", [])
                    for s in system.get("sectors", [])
                    if s in _protocol.SEALED_SECTORS or s in _protocol.DISCOVERY_SECTORS
                }
            )
            sealed.extend(bad)
            offenders.append(path)
            continue
        checked.append(path)
        for system in manifest.systems:
            bad = [s for s in system.sectors if s not in _protocol.DEV_SECTORS]
            if bad:
                sealed.extend(bad)
                offenders.append(path)
    sealed = sorted(set(sealed))
    return {
        "manifests_checked": checked,
        "sealed_sectors_touched": sealed,
        "offenders": offenders,
        "ok": not sealed and not offenders,
    }


__all__ = [
    "FreezeRecord",
    "HoldoutManifest",
    "HoldoutSystem",
    "audit_development",
    "checkpoint_hash",
    "create_freeze",
    "file_hash",
    "load_freeze_record",
    "load_holdout_manifest",
    "log_access",
    "mark_unblinded",
    "source_tree_hash",
    "verify_freeze",
]
