"""Binary-aware event handling for the bounded Survey B pilot.

This module deliberately does not alter the frozen single-star matcher. A
circumbinary transit may occur at different binary phases and need not obey a
constant-period event sequence, so the pilot only masks the binary's own
primary and secondary eclipses and ranks retained events without a period
plausibility gate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tess_assoc import protocol as _protocol
from tess_assoc._validate import (
    is_finite_number,
    is_strict_int,
    require_finite,
    require_positive_finite,
    require_strict_int,
)
from tess_assoc.event import EventRecord
from tess_assoc.extract import BTJD_OFFSET


PILOT_VERSION = "survey-b-v1"
PRODUCT = "TESS-SPOC FFI"
TARGET_CLASSES = frozenset({"circumbinary-positive", "eclipsing-binary-control"})


@dataclass(frozen=True)
class BinaryPilotTarget:
    """One eclipsing binary in the frozen Survey B pilot manifest."""

    tic_id: int
    name: str
    period_days: float
    t0_bjd_tdb: float
    primary_duration_days: float
    secondary_phase: float
    sectors: tuple[int, ...]
    target_class: str
    secondary_duration_days: float | None = None

    def __post_init__(self) -> None:
        require_strict_int("tic_id", self.tic_id, minimum=1)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("binary target name must be non-empty")
        require_positive_finite("period_days", self.period_days)
        require_finite("t0_bjd_tdb", self.t0_bjd_tdb)
        require_positive_finite("primary_duration_days", self.primary_duration_days)
        if not is_finite_number(self.secondary_phase) or not 0 < self.secondary_phase < 1:
            raise ValueError("secondary_phase must be finite and between 0 and 1")
        if self.secondary_duration_days is not None:
            require_positive_finite(
                "secondary_duration_days", self.secondary_duration_days
            )
        if self.target_class not in TARGET_CLASSES:
            raise ValueError(f"target_class must be one of {sorted(TARGET_CLASSES)}")
        if not isinstance(self.sectors, (list, tuple)) or not self.sectors:
            raise ValueError("binary target sectors must be non-empty")
        if any(
            not is_strict_int(sector) or sector not in _protocol.DEV_SECTORS
            for sector in self.sectors
        ):
            _protocol.validate_no_temporal_leak(set(self.sectors))
            raise ValueError("Survey B pilot targets must use development sectors 1-79")
        if tuple(sorted(set(self.sectors))) != tuple(self.sectors):
            raise ValueError("binary target sectors must be sorted and unique")
        object.__setattr__(self, "sectors", tuple(self.sectors))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by the pilot manifest."""
        result = {
            "tic_id": self.tic_id,
            "name": self.name,
            "period_days": self.period_days,
            "t0_bjd_tdb": self.t0_bjd_tdb,
            "primary_duration_days": self.primary_duration_days,
            "secondary_phase": self.secondary_phase,
            "secondary_duration_days": self.secondary_duration_days,
            "sectors": list(self.sectors),
            "target_class": self.target_class,
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BinaryPilotTarget":
        required = (
            "tic_id",
            "name",
            "period_days",
            "t0_bjd_tdb",
            "primary_duration_days",
            "secondary_phase",
            "sectors",
            "target_class",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"binary target missing keys: {missing}")
        return cls(
            tic_id=value["tic_id"],
            name=value["name"],
            period_days=value["period_days"],
            t0_bjd_tdb=value["t0_bjd_tdb"],
            primary_duration_days=value["primary_duration_days"],
            secondary_phase=value["secondary_phase"],
            secondary_duration_days=value.get("secondary_duration_days"),
            sectors=tuple(value["sectors"]),
            target_class=value["target_class"],
        )


def build_binary_pilot_manifest(
    name: str,
    targets: Sequence[BinaryPilotTarget],
    *,
    source_catalog: str,
) -> dict[str, Any]:
    """Build a versioned, JSON-serializable Survey B target manifest."""
    if not isinstance(name, str) or not name:
        raise ValueError("pilot manifest name must be non-empty")
    if not isinstance(source_catalog, str) or not source_catalog:
        raise ValueError("source_catalog must be non-empty")
    if not targets or not all(isinstance(target, BinaryPilotTarget) for target in targets):
        raise ValueError("pilot manifest needs BinaryPilotTarget records")
    if len({target.tic_id for target in targets}) != len(targets):
        raise ValueError("pilot target TIC IDs must be unique")
    if len({target.name for target in targets}) != len(targets):
        raise ValueError("pilot target names must be unique")
    return {
        "name": name,
        "version": PILOT_VERSION,
        "product": PRODUCT,
        "source_catalog": source_catalog,
        "targets": [target.to_dict() for target in targets],
    }


def load_binary_pilot_manifest(
    payload: Mapping[str, Any] | str,
) -> tuple[BinaryPilotTarget, ...]:
    """Load and validate a Survey B manifest from a mapping or JSON path."""
    if isinstance(payload, str):
        payload = json.loads(Path(payload).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("pilot manifest must be a mapping")
    for key in ("name", "version", "product", "source_catalog", "targets"):
        if key not in payload:
            raise ValueError(f"pilot manifest missing key: {key}")
    if payload["version"] != PILOT_VERSION:
        raise ValueError(f"unsupported Survey B manifest version: {payload['version']}")
    if payload["product"] != PRODUCT:
        raise ValueError(f"pilot manifest product must be {PRODUCT}")
    if not isinstance(payload["source_catalog"], str) or not payload["source_catalog"]:
        raise ValueError("source_catalog must be non-empty")
    if not isinstance(payload["targets"], list) or not payload["targets"]:
        raise ValueError("pilot manifest targets must be a non-empty list")
    targets = tuple(BinaryPilotTarget.from_dict(value) for value in payload["targets"])
    if len({target.tic_id for target in targets}) != len(targets):
        raise ValueError("pilot target TIC IDs must be unique")
    if len({target.name for target in targets}) != len(targets):
        raise ValueError("pilot target names must be unique")
    return targets


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def eclipse_windows(
    target: BinaryPilotTarget,
    coverage_windows: Mapping[int, Sequence[tuple[float, float]]],
    *,
    buffer_days: float = 0.05,
) -> dict[int, list[tuple[float, float]]]:
    """Return primary and secondary eclipse masks intersecting TESS coverage."""
    require_positive_finite("buffer_days", buffer_days)
    t0_btjd = target.t0_bjd_tdb - BTJD_OFFSET
    masks: dict[int, list[tuple[float, float]]] = {}
    secondary_duration = target.secondary_duration_days or target.primary_duration_days
    for sector in target.sectors:
        spans = coverage_windows.get(sector)
        if not spans:
            continue
        sector_masks: list[tuple[float, float]] = []
        for start, end in spans:
            require_finite("coverage window start", start)
            require_finite("coverage window end", end)
            if end <= start:
                raise ValueError("coverage window end must be after start")
            for phase, duration in (
                (0.0, target.primary_duration_days),
                (target.secondary_phase, secondary_duration),
            ):
                first = math.floor((start - t0_btjd) / target.period_days) - 1
                last = math.ceil((end - t0_btjd) / target.period_days) + 1
                half_width = duration / 2.0 + buffer_days
                for cycle in range(first, last + 1):
                    center = t0_btjd + (cycle + phase) * target.period_days
                    masked_start = max(start, center - half_width)
                    masked_end = min(end, center + half_width)
                    if masked_end > masked_start:
                        sector_masks.append((masked_start, masked_end))
        if sector_masks:
            masks[sector] = _merge_windows(sector_masks)
    return masks


def _value(event: EventRecord | Mapping[str, Any], key: str) -> Any:
    if isinstance(event, EventRecord):
        return getattr(event, key)
    try:
        return event[key]
    except KeyError as error:
        raise ValueError(f"binary pilot event missing key: {key}") from error


def _event_summary(
    event: EventRecord | Mapping[str, Any], index: int, target: BinaryPilotTarget
) -> dict[str, Any]:
    t0 = float(_value(event, "t0"))
    phase = (
        (t0 - (target.t0_bjd_tdb - BTJD_OFFSET)) / target.period_days
    ) % 1.0
    quality = _value(event, "quality")
    truth = quality.get("pilot_truth", "") if isinstance(quality, Mapping) else ""
    event_id = quality.get("event_id") if isinstance(quality, Mapping) else None
    return {
        "event_index": index,
        "event_id": event_id or f"event-{index}",
        "tic_id": int(_value(event, "tic_id")),
        "sector": int(_value(event, "sector")),
        "t0": t0,
        "depth": float(_value(event, "depth")),
        "duration_days": float(_value(event, "duration_days")),
        "snr": float(_value(event, "snr")),
        "binary_phase": phase,
        "pilot_truth": str(truth),
    }


def _overlaps_mask(
    t0: float, duration_days: float, masks: Sequence[tuple[float, float]]
) -> bool:
    event_start = t0 - duration_days / 2.0
    event_end = t0 + duration_days / 2.0
    return any(event_start <= end and event_end >= start for start, end in masks)


def _inside_coverage(
    t0: float, duration_days: float, spans: Sequence[tuple[float, float]]
) -> bool:
    event_start = t0 - duration_days / 2.0
    event_end = t0 + duration_days / 2.0
    return any(start <= event_start and event_end <= end for start, end in spans)


def _validated_event(
    event: EventRecord | Mapping[str, Any],
) -> EventRecord:
    if isinstance(event, EventRecord):
        return event
    if isinstance(event, Mapping):
        return EventRecord.from_dict(dict(event))
    raise ValueError("binary pilot events must be EventRecord records or mappings")


def _rank_pairs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, first in enumerate(events):
        for second in events[index + 1 :]:
            if first["sector"] == second["sector"]:
                continue
            truth = (
                first["pilot_truth"] == "same-circumbinary"
                and second["pilot_truth"] == "same-circumbinary"
            )
            pairs.append(
                {
                    "event_a": first["event_id"],
                    "event_b": second["event_id"],
                    "sectors": [first["sector"], second["sector"]],
                    "t0s": [first["t0"], second["t0"]],
                    "score": min(first["snr"], second["snr"]),
                    "binary_phase": [first["binary_phase"], second["binary_phase"]],
                    "timing_model": "binary-phase-free",
                    "period_constraint": "not-applied",
                    "truth": truth,
                }
            )
    return sorted(pairs, key=lambda pair: (-pair["score"], pair["t0s"]))


def run_binary_pilot(
    targets: Sequence[BinaryPilotTarget],
    events: Sequence[EventRecord | Mapping[str, Any]],
    coverage_windows: Mapping[int, Sequence[tuple[float, float]]],
    *,
    min_snr: float = 7.0,
    mask_buffer_days: float = 0.05,
) -> dict[str, Any]:
    """Mask EB eclipses and rank cross-sector event pairs without alias gating."""
    require_positive_finite("min_snr", min_snr)
    if len({target.tic_id for target in targets}) != len(targets):
        raise ValueError("pilot target TIC IDs must be unique")
    if len({target.name for target in targets}) != len(targets):
        raise ValueError("pilot target names must be unique")
    event_groups: dict[int, list[tuple[int, EventRecord | Mapping[str, Any]]]] = {}
    for index, event in enumerate(events):
        validated = _validated_event(event)
        event_groups.setdefault(validated.tic_id, []).append((index, validated))

    systems: dict[str, Any] = {}
    for target in targets:
        masks = eclipse_windows(
            target, coverage_windows, buffer_days=mask_buffer_days
        )
        input_events = event_groups.get(target.tic_id, [])
        retained: list[dict[str, Any]] = []
        masked = 0
        below_snr = 0
        out_of_cohort = 0
        for index, event in input_events:
            sector = int(_value(event, "sector"))
            t0 = float(_value(event, "t0"))
            if sector not in target.sectors or sector not in coverage_windows:
                out_of_cohort += 1
                continue
            duration_days = float(_value(event, "duration_days"))
            spans = coverage_windows[sector]
            if not _inside_coverage(t0, duration_days, spans):
                out_of_cohort += 1
                continue
            if _overlaps_mask(t0, duration_days, masks.get(sector, ())):
                masked += 1
                continue
            snr = float(_value(event, "snr"))
            require_finite("event snr", snr)
            if snr < min_snr:
                below_snr += 1
                continue
            retained.append(_event_summary(event, index, target))
        event_ids = [event["event_id"] for event in retained]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError(f"duplicate event IDs for binary target {target.name}")
        pairs = _rank_pairs(retained)
        systems[target.name] = {
            "tic_id": target.tic_id,
            "target_class": target.target_class,
            "status": "complete",
            "n_input_events": len(input_events),
            "n_masked_events": masked,
            "n_below_snr": below_snr,
            "n_out_of_cohort": out_of_cohort,
            "n_retained_events": len(retained),
            "n_pairs_ranked": len(pairs),
            "eclipse_masks": {str(sector): windows for sector, windows in masks.items()},
            "events": retained,
            "pairs": pairs,
        }

    positive_targets = [
        system for system in systems.values()
        if system["target_class"] == "circumbinary-positive"
    ]
    recovered = sum(any(pair["truth"] for pair in system["pairs"]) for system in positive_targets)
    return {
        "pilot_version": PILOT_VERSION,
        "status": "complete",
        "min_snr": min_snr,
        "mask_buffer_days": mask_buffer_days,
        "timing_model": "binary-phase-free",
        "systems": systems,
        "metrics": {
            "positive_targets": len(positive_targets),
            "positive_targets_with_truth_pair": recovered,
            "control_targets_with_ranked_pair": sum(
                bool(system["pairs"])
                for system in systems.values()
                if system["target_class"] == "eclipsing-binary-control"
            ),
        },
    }


def render_binary_pilot_report(results: Mapping[str, Any]) -> str:
    """Render a concise, auditable report without implying planet confirmation."""
    lines = [
        f"# Survey B pilot ({results['pilot_version']})",
        f"Status: {results['status']}; timing model: {results['timing_model']}.",
        "",
        "## Metrics",
    ]
    for key, value in results["metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Systems"])
    for name, system in results["systems"].items():
        lines.append(
            f"- {name} (TIC {system['tic_id']}): "
            f"{system['n_input_events']} input, {system['n_masked_events']} masked, "
            f"{system['n_retained_events']} retained, "
            f"{system['n_pairs_ranked']} cross-sector pair(s)."
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "BinaryPilotTarget",
    "PILOT_VERSION",
    "build_binary_pilot_manifest",
    "eclipse_windows",
    "load_binary_pilot_manifest",
    "render_binary_pilot_report",
    "run_binary_pilot",
]
