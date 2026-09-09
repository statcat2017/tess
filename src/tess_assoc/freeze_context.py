"""Authenticated context for one frozen cohort run."""

from __future__ import annotations

import json
import os
import tempfile
import dataclasses
import fcntl
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tess_assoc.freeze import (
    FreezeRecord,
    _read_authenticated_manifest,
    _utcnow,
    check_frozen_system,
    load_freeze_record,
    verify_freeze,
)

CohortKey = Literal["holdout", "discovery"]
_CONTEXT_TOKEN = object()


def _canonical_system(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("frozen cohort systems must be objects")
    normalized = dict(payload)
    normalized.setdefault("toi", "")
    normalized.setdefault("period_days", None)
    normalized.setdefault("t0_bjd_tdb", None)
    normalized.setdefault("duration_hours", None)
    normalized.setdefault("known_planets", [])
    if not isinstance(normalized["sectors"], (list, tuple)):
        raise ValueError("frozen system sectors must be a list")
    normalized["sectors"] = list(normalized["sectors"])
    if not isinstance(normalized["known_planets"], list):
        raise ValueError("known_planets must be a list")
    normalized["known_planets"] = [
        {
            key: planet.get(key, "" if key == "disposition" else "manifest")
            for key in (
                "name", "period_days", "t0_bjd_tdb", "duration_days",
                "source", "disposition",
            )
        }
        for planet in normalized["known_planets"]
        if isinstance(planet, dict)
    ]
    if len(normalized["known_planets"]) != len(payload.get("known_planets", ())):
        raise ValueError("frozen known planets must be objects")
    return normalized


def _system_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(
        _canonical_system(payload), sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True, init=False)
class FrozenRunContext:
    """Verified capability for one immutable cohort manifest and freeze."""

    record: FreezeRecord
    freeze_path: str
    manifest_path: str
    cohort_key: CohortKey
    system_payloads: MappingProxyType

    def __init__(
        self,
        record: FreezeRecord,
        freeze_path: str,
        manifest_path: str,
        cohort_key: CohortKey,
        system_payloads: MappingProxyType,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _CONTEXT_TOKEN:
            raise TypeError("FrozenRunContext must be opened with FrozenRunContext.open")
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "freeze_path", freeze_path)
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "cohort_key", cohort_key)
        object.__setattr__(self, "system_payloads", system_payloads)

    @classmethod
    def open(
        cls,
        freeze_path: str,
        manifest_path: str,
        config: Any,
        *,
        cohort_key: CohortKey,
    ) -> "FrozenRunContext":
        if cohort_key not in ("holdout", "discovery"):
            raise ValueError("cohort_key must be 'holdout' or 'discovery'")
        record = verify_freeze(freeze_path, config)
        payload = _read_authenticated_manifest(manifest_path, record, cohort_key)
        systems = payload.get("systems")
        if not isinstance(systems, list):
            raise ValueError("frozen cohort manifest systems must be a list")
        fingerprints: dict[int, str] = {}
        for system in systems:
            if not isinstance(system, dict) or not isinstance(system.get("tic_id"), int):
                raise ValueError("frozen cohort systems must have integer TIC ids")
            tic_id = system["tic_id"]
            if tic_id in fingerprints:
                raise ValueError(f"TIC {tic_id} is duplicated in frozen manifest")
            fingerprints[tic_id] = _system_fingerprint(system)
        return cls(
            record,
            str(Path(freeze_path).resolve()),
            str(Path(manifest_path).resolve()),
            cohort_key,
            MappingProxyType(fingerprints),
            _token=_CONTEXT_TOKEN,
        )

    def read_manifest(self, path: str) -> dict[str, Any]:
        """Read a manifest after authenticating the exact bytes parsed."""
        return _read_authenticated_manifest(path, self.record, self.cohort_key)

    def check_thresholds(self, thresholds: dict[str, float], label: str) -> None:
        if dict(thresholds) != self.record.thresholds:
            raise ValueError(f"{label} thresholds differ from frozen thresholds")

    def check_system(
        self,
        tic_id: int,
        sectors: set[int],
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        check_frozen_system(self.record, self.cohort_key, tic_id, sectors)
        if payload is not None:
            expected = _system_fingerprint(payload)
            if self.system_payloads.get(tic_id) != expected:
                raise ValueError(f"TIC {tic_id} payload differs from frozen manifest")

    def require_unblinded(self) -> None:
        if self.record.unblinded_utc is None:
            raise ValueError("frozen context has not been unblinded")

    def mark_unblinded(self) -> "FrozenRunContext":
        stamped = self.record.stamped(_utcnow())
        if stamped is not self.record:
            with open(self.freeze_path + ".lock", "a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                current = load_freeze_record(self.freeze_path)
                if current.unblinded_utc is not None:
                    if dataclasses.replace(
                        current, unblinded_utc=None
                    ) != dataclasses.replace(self.record, unblinded_utc=None):
                        raise ValueError("freeze record changed after context opened")
                    stamped = current
                else:
                    if current != self.record:
                        raise ValueError("freeze record changed after context opened")
                    directory = str(Path(self.freeze_path).parent)
                    fd, temporary = tempfile.mkstemp(
                        dir=directory, prefix=".freeze-", text=True
                    )
                    try:
                        with os.fdopen(fd, "w") as handle:
                            json.dump(stamped.to_dict(), handle, indent=2)
                            handle.write("\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, self.freeze_path)
                        directory_fd = os.open(directory, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except BaseException:
                        try:
                            os.unlink(temporary)
                        except FileNotFoundError:
                            pass
                        raise
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return type(self)(
            stamped,
            self.freeze_path,
            self.manifest_path,
            self.cohort_key,
            self.system_payloads,
            _token=_CONTEXT_TOKEN,
        )

    def evidence(self) -> dict[str, str | None]:
        return {
            "code_sha": self.record.code_sha,
            "created_utc": self.record.created_utc,
            "unblinded_utc": self.record.unblinded_utc,
        }
