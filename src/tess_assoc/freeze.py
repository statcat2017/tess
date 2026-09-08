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
from tess_assoc.matcher import validate_matcher_thresholds


class _FrozenDict(dict):
    """JSON-compatible dict that rejects in-place mutation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("freeze record payloads are immutable")

    __delitem__ = __setitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


class _FrozenList(list):
    """JSON-compatible list that rejects in-place mutation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("freeze record payloads are immutable")

    __delitem__ = __iadd__ = __imul__ = __setitem__ = _immutable
    append = clear = extend = insert = pop = remove = reverse = sort = _immutable


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_payload(item) for item in value)
    if isinstance(value, tuple):
        return _FrozenList(_freeze_payload(item) for item in value)
    return value


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
            if sector not in _protocol.DEV_SECTORS | _protocol.SEALED_SECTORS:
                raise ValueError("holdout sectors must be development or sealed sectors")
        if len(set(self.sectors)) != len(self.sectors):
            raise ValueError("system sectors must be unique")
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
    source_sha256: str | None = field(default=None, repr=False, compare=False)

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
        validate_matcher_thresholds(self.matcher_thresholds)
        if not isinstance(self.systems, (list, tuple)) or not self.systems:
            raise ValueError("systems must be a non-empty list")
        if not all(isinstance(s, HoldoutSystem) for s in self.systems):
            raise ValueError("systems must be HoldoutSystem records")
        if len({s.name for s in self.systems}) != len(self.systems):
            raise ValueError("system names must be unique")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a SHA-256 hex string")
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
    system_sectors: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_version",
            "code_sha",
            "created_utc",
            "ablation",
            "ephemeris_source",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty str")
        if self.unblinded_utc is not None and not isinstance(self.unblinded_utc, str):
            raise ValueError("unblinded_utc must be a str or None")
        if not isinstance(self.thresholds, dict):
            raise ValueError("thresholds must be a dict")
        validate_matcher_thresholds(self.thresholds)
        if not isinstance(self.learn_config, dict):
            raise ValueError("learn_config must be a dict")
        if not isinstance(self.injection, dict):
            raise ValueError("injection must be a dict")
        if not isinstance(self.manifests, dict):
            raise ValueError("manifests must be a dict")
        for key, value in self.manifests.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, dict)
                or not isinstance(value.get("path"), str)
                or not value["path"]
                or not isinstance(value.get("sha256"), str)
                or len(value["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in value["sha256"])
            ):
                raise ValueError("manifests must map names to path/hash dicts")
        if not isinstance(self.systems, dict):
            raise ValueError("systems must be a dict")
        for key, sectors in self.systems.items():
            if not isinstance(key, str) or not isinstance(sectors, (list, tuple)):
                raise ValueError("systems must map strings to sector lists")
            for sector in sectors:
                require_strict_int("system sector", sector, minimum=1)
        if not isinstance(self.system_sectors, dict):
            raise ValueError("system_sectors must be a dict")
        for cohort, mapping in self.system_sectors.items():
            if not isinstance(cohort, str) or not isinstance(mapping, dict):
                raise ValueError("system_sectors must map cohorts to TIC maps")
            for tic, sectors in mapping.items():
                if not isinstance(tic, str) or not tic.isdigit():
                    raise ValueError("system_sectors TIC keys must be strings")
                require_strict_int("system TIC", int(tic), minimum=1)
                if not isinstance(sectors, (list, tuple)) or not sectors:
                    raise ValueError("system_sectors must map TICs to sector lists")
                for sector in sectors:
                    require_strict_int("system sector", sector, minimum=1)
                if len(set(sectors)) != len(sectors):
                    raise ValueError("system_sectors sector lists must be unique")
        if set(self.system_sectors) != set(self.systems):
            raise ValueError("system and system_sectors cohort keys differ")
        if self.checkpoint_sha is not None and not isinstance(
            self.checkpoint_sha, str
        ):
            raise ValueError("checkpoint_sha must be a str or None")
        require_positive_finite(
            "proposer_snr_threshold", self.proposer_snr_threshold
        )
        try:
            thresholds = dict(self.thresholds)
            learn_config = _canonical(self.learn_config)
            injection = _canonical(self.injection)
            manifests = _canonical(self.manifests)
            systems = _canonical(self.systems)
            system_sectors = _canonical(self.system_sectors)
        except (TypeError, ValueError) as e:
            raise ValueError("freeze record payloads must be JSON-compatible") from e
        object.__setattr__(self, "thresholds", _freeze_payload(thresholds))
        object.__setattr__(self, "learn_config", _freeze_payload(learn_config))
        object.__setattr__(self, "injection", _freeze_payload(injection))
        object.__setattr__(self, "manifests", _freeze_payload(manifests))
        object.__setattr__(self, "systems", _freeze_payload(systems))
        object.__setattr__(self, "system_sectors", _freeze_payload(system_sectors))

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
            "system_sectors": _canonical(self.system_sectors),
            "checkpoint_sha": self.checkpoint_sha,
            "ephemeris_source": self.ephemeris_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FreezeRecord":
        if not isinstance(d, dict):
            raise ValueError("freeze record must be a dict")
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
            thresholds=d["thresholds"],
            learn_config=d["learn_config"],
            ablation=d["ablation"],
            proposer_snr_threshold=d["proposer_snr_threshold"],
            injection=d["injection"],
            manifests=d["manifests"],
            systems=d["systems"],
            checkpoint_sha=d.get("checkpoint_sha"),
            ephemeris_source=d["ephemeris_source"],
            system_sectors=d.get("system_sectors", {}),
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
    return json.loads(json.dumps(obj, allow_nan=False))


def check_manifest_bytes(path: str, record: FreezeRecord, key: str) -> None:
    """Byte-binding for any gated cohort manifest (holdout, discovery, ...)."""
    info = record.manifests.get(key, {})
    if not info or "sha256" not in info:
        raise ValueError(f"freeze record pins no {key} manifest")
    if file_hash(path) != info["sha256"]:
        raise ValueError(f"{key} manifest bytes differ from frozen manifest")


def check_frozen_system(
    record: FreezeRecord,
    cohort_key: str,
    tic_id: int,
    sectors: set[int],
) -> None:
    """Bind one frozen replay to its pinned TIC and sector assignment."""
    cohort_systems = record.system_sectors.get(cohort_key, {})
    if not isinstance(cohort_systems, dict):
        raise ValueError(f"freeze record has no pinned {cohort_key} manifest")
    pinned_sectors = cohort_systems.get(str(tic_id))
    if pinned_sectors is None:
        raise ValueError(f"TIC {tic_id} is not uniquely pinned in {cohort_key}")
    if not isinstance(pinned_sectors, list):
        raise ValueError(f"TIC {tic_id} sectors differ from frozen {cohort_key} system")
    try:
        unique_sector_count = len(set(pinned_sectors))
    except TypeError as e:
        raise ValueError(
            f"TIC {tic_id} sectors differ from frozen {cohort_key} system"
        ) from e
    if len(pinned_sectors) != unique_sector_count:
        raise ValueError(f"TIC {tic_id} sectors differ from frozen {cohort_key} system")
    for sector in pinned_sectors:
        require_strict_int("frozen sector", sector, minimum=1)
        if sector not in _protocol.ALL_KNOWN_SECTORS:
            raise ValueError(f"TIC {tic_id} sectors differ from frozen {cohort_key} system")
    try:
        sectors_match = set(pinned_sectors) == sectors
    except TypeError as e:
        raise ValueError(
            f"TIC {tic_id} sectors differ from frozen {cohort_key} system"
        ) from e
    if not sectors_match:
        raise ValueError(f"TIC {tic_id} sectors differ from frozen {cohort_key} system")


def create_freeze(
    dev_manifest_path: str,
    cohort_manifest_path: str,
    config,
    *,
    output_path: str,
    ablation: str = "morphology+scalars",
    checkpoint_sha: str | None = None,
    cohort_key: str = "holdout",
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
    with open(cohort_manifest_path) as f:
        cohort_raw = json.load(f)
    if not isinstance(cohort_raw, dict):
        raise ValueError("cohort manifest must be a dict")
    if not isinstance(cohort_raw.get("systems"), list) or not cohort_raw["systems"]:
        raise ValueError("cohort manifest must hold a non-empty systems list")
    if cohort_key not in ("holdout", "discovery"):
        raise ValueError("cohort_key must be 'holdout' or 'discovery'")
    allowed_sectors = (
        _protocol.DEV_SECTORS | _protocol.SEALED_SECTORS
        if cohort_key == "holdout"
        else _protocol.DEV_SECTORS | _protocol.DISCOVERY_SECTORS
    )
    cohort_systems = []
    for system in cohort_raw["systems"]:
        if not isinstance(system, dict):
            raise ValueError("cohort systems must be dicts")
        for key in ("name", "tic_id", "sectors"):
            if key not in system:
                raise ValueError(f"cohort system missing key: {key}")
        require_strict_int("cohort system TIC", system["tic_id"], minimum=1)
        if not isinstance(system["sectors"], list) or not system["sectors"]:
            raise ValueError("cohort system sectors must be a non-empty list")
        for sector in system["sectors"]:
            require_strict_int("cohort sector", sector, minimum=1)
            if sector not in allowed_sectors:
                raise ValueError(f"sector {sector} is not allowed in {cohort_key} cohort")
        if len(set(system["sectors"])) != len(system["sectors"]):
            raise ValueError("cohort system sectors must be unique")
        cohort_systems.append(system["tic_id"])
    if len(set(cohort_systems)) != len(cohort_systems):
        raise ValueError("cohort system TICs must be unique")
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
            cohort_key: {
                "path": str(Path(cohort_manifest_path).resolve()),
                "sha256": file_hash(cohort_manifest_path),
            },
        },
        systems={
            "dev": sorted(s.tic_id for s in dev.systems),
            cohort_key: sorted(cohort_systems),
        },
        system_sectors={
            "dev": {
                str(s.tic_id): list(s.sectors)
                for s in dev.systems
            },
            cohort_key: {
                str(system["tic_id"]): list(system["sectors"])
                for system in cohort_raw["systems"]
            },
        },
        checkpoint_sha=checkpoint_sha,
        ephemeris_source=dev.ephemeris_source,
    )
    Path(output_path).write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return record


def load_freeze_record(path: str) -> FreezeRecord:
    with open(path) as f:
        return FreezeRecord.from_dict(json.load(f))


def _manifest_system_sectors(raw: Any) -> dict[str, list[int]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("systems"), list):
        raise ValueError("manifest systems are malformed")
    result: dict[str, list[int]] = {}
    for system in raw["systems"]:
        if not isinstance(system, dict):
            raise ValueError("manifest systems are malformed")
        tic_id = system.get("tic_id")
        sectors = system.get("sectors")
        require_strict_int("system TIC", tic_id, minimum=1)
        if not isinstance(sectors, list) or not sectors:
            raise ValueError("manifest systems are malformed")
        for sector in sectors:
            require_strict_int("system sector", sector, minimum=1)
        key = str(tic_id)
        if key in result or len(set(sectors)) != len(sectors):
            raise ValueError("manifest systems are malformed")
        result[key] = list(sectors)
    return result


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
    if not record.system_sectors:
        problems.append("freeze record has no system sector map")
    dev_info = record.manifests.get("dev", {})
    if not dev_info:
        problems.append("dev manifest not pinned by freeze")
    elif Path(dev_info["path"]).exists():
        if file_hash(dev_info["path"]) != dev_info["sha256"]:
            problems.append("dev manifest changed since freeze")
        else:
            from tess_assoc.replay import load_replay_manifest

            dev = load_replay_manifest(dev_info["path"])
            if dict(dev.matcher_thresholds) != record.thresholds:
                problems.append("matcher thresholds changed since freeze")
            actual = _manifest_system_sectors(
                {"systems": [
                    {"tic_id": s.tic_id, "sectors": list(s.sectors)}
                    for s in dev.systems
                ]}
            )
            if actual != record.system_sectors.get("dev"):
                problems.append("dev system sector map changed since freeze")
            if sorted(actual) != [str(tic) for tic in record.systems.get("dev", [])]:
                problems.append("dev system TIC list changed since freeze")
    else:
        # The freeze stores the canonical dev map; the source file may move.
        pass
    cohort_keys = [k for k in record.manifests if k != "dev"]
    if set(record.systems) != set(record.manifests):
        problems.append("freeze manifest and system cohort keys differ")
    if not cohort_keys or any(
        "sha256" not in record.manifests[k] for k in cohort_keys
    ):
        problems.append("cohort manifest not pinned by freeze")
    for key in cohort_keys:
        info = record.manifests.get(key, {})
        if not Path(info["path"]).exists():
            continue
        if file_hash(info["path"]) != info["sha256"]:
            problems.append(f"{key} manifest changed since freeze")
            continue
        try:
            with open(info["path"]) as f:
                raw = json.load(f)
            actual_map = _manifest_system_sectors(raw)
        except (KeyError, OSError, TypeError, ValueError):
            problems.append(f"{key} manifest systems are malformed")
        else:
            actual_systems = sorted(int(tic) for tic in actual_map)
            if actual_map != record.system_sectors.get(key):
                problems.append(f"{key} system sector map changed since freeze")
            if actual_systems != record.systems.get(key, []):
                problems.append(f"{key} system TIC list changed since freeze")
    if record.learn_config != _canonical(dataclasses.asdict(config)):
        problems.append("learn config changed since freeze")
    if problems:
        raise ValueError("freeze verification failed: " + "; ".join(problems))
    return record


def _parse_holdout_manifest(d: dict[str, Any]) -> HoldoutManifest:
    if not isinstance(d, dict):
        raise ValueError("holdout manifest must be a dict")
    for key in (
        "name", "product", "ephemeris_source", "epoch_match_tol_days",
        "window_half_span_days", "resample_samples", "matcher_thresholds",
        "systems",
    ):
        if key not in d:
            raise ValueError(f"holdout manifest missing key: {key}")
    if not isinstance(d["systems"], list) or not d["systems"]:
        raise ValueError("holdout systems must be a non-empty list")
    if not isinstance(d["matcher_thresholds"], dict):
        raise ValueError("holdout matcher_thresholds must be a dict")
    required_system_keys = {
        "name", "tic_id", "period_days", "t0_bjd_tdb", "duration_hours", "sectors"
    }
    for system in d["systems"]:
        if not isinstance(system, dict):
            raise ValueError("each holdout system must be a dict")
        missing = required_system_keys - set(system)
        if missing:
            raise ValueError(f"holdout system missing key: {sorted(missing)[0]}")
        if not isinstance(system["sectors"], (list, tuple)):
            raise ValueError("holdout system sectors must be a list")
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
    check_manifest_bytes(path, record, "holdout")
    with open(path) as f:
        manifest = _parse_holdout_manifest(json.load(f))
    if dict(manifest.matcher_thresholds) != record.thresholds:
        raise ValueError("holdout thresholds differ from frozen thresholds")
    object.__setattr__(manifest, "source_sha256", record.manifests["holdout"]["sha256"])
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
    "check_manifest_bytes",
    "checkpoint_hash",
    "check_frozen_system",
    "create_freeze",
    "file_hash",
    "load_freeze_record",
    "load_holdout_manifest",
    "log_access",
    "mark_unblinded",
    "source_tree_hash",
    "verify_freeze",
]
