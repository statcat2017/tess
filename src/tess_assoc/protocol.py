"""Phase 0 protocol freeze (v1).

Source of truth for frozen scientific definitions.
Derived from RESEARCH_PIPELINE_PRD.md and RESEARCH_PROPOSAL.md.

Any change to these values requires a new protocol version;
v1 values are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

PROTOCOL_VERSION = "v1"
PROTOCOL_SOURCE_DOCS = (
    "RESEARCH_PIPELINE_PRD.md",
    "RESEARCH_PROPOSAL.md",
)

# --- Photometric product (PRD Implementation Decisions) ---
PRIMARY_PHOTOMETRIC_PRODUCT = "TESS-SPOC FFI"
RESERVED_FOR_VETTING_ONLY = ("SPOC", "QLP", "TGLC")

# --- Sector boundaries (PRD + Proposal §21) ---
DEV_SECTORS: FrozenSet[int] = frozenset(range(1, 80))  # 1–79 inclusive
SEALED_SECTORS: FrozenSet[int] = frozenset(range(80, 106))  # 80–105 inclusive
DISCOVERY_SECTORS: FrozenSet[int] = frozenset({106})
ALL_KNOWN_SECTORS: FrozenSet[int] = DEV_SECTORS | SEALED_SECTORS | DISCOVERY_SECTORS

# --- Orbital inference (Proposal §19) ---
LONG_PERIOD_LOWER_BOUND_DAYS = 27.0
ALIAS_FORMULA = "P_n = DeltaT / n"

# --- Event record contract (PRD User Stories 3-5) ---
EVENT_REQUIRED_FIELDS: tuple[str, ...] = (
    "tic_id",
    "sector",
    "t0",
    "local_time",
    "local_flux",
    "depth",
    "duration_days",
    "snr",
    "stellar_meta",
    "quality",
)
EVENT_OPTIONAL_FIELDS: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "background_flux",
    "crowding",
)

# TIC is grouping/partition key only, never a model feature.
TIC_AS_FEATURE_FORBIDDEN = True
MODEL_FEATURE_BLACKLIST: tuple[str, ...] = ("tic_id",)

# --- Association semantics (PRD Implementation Decisions) ---
LEARNED_OUTPUT_SEMANTICS = "P(same transit-producing object)"
LEARNED_OUTPUT_IS_NOT = (
    "P(star contains a planet)",
    "orbital-period prediction",
)
PRIMARY_MODEL_INPUT = "normalized local transit morphology"
OPTIONAL_MODEL_INPUTS: tuple[str, ...] = (
    "depth",
    "duration_days",
    "snr",
    "stellar_meta",
    "crowding",
)
DETERMINISTIC_BASELINE_COMPONENTS: tuple[str, ...] = (
    "relative depth difference",
    "relative duration difference",
    "normalized morphology correlation",
    "timing plausibility",
)

# --- Metrics (PRD Implementation Decisions + Proposal §23) ---
PRIMARY_METRIC = "true-repeat retrieval at fixed candidate burden"
SUPPORTING_METRICS: tuple[str, ...] = (
    "precision-recall AUC",
    "top-1 retrieval",
    "top-5 retrieval",
    "mean reciprocal rank",
    "recall at fixed false-association rate",
    "precision at fixed human-vetting budget",
)
FULL_PIPELINE_REPORT_FIELDS: tuple[str, ...] = (
    "recovered known long-period systems",
    "retrieved true repeat events",
    "candidate pairs per recovered system",
    "aliases before window filtering",
    "aliases after window filtering",
    "false associations per 1000 targets",
)

# --- Reproducibility metadata (PRD Implementation Decisions) ---
REPRODUCIBILITY_FIELDS: tuple[str, ...] = (
    "tic_list",
    "sector_manifest",
    "source_product_version",
    "catalogue_version",
    "catalogue_download_date",
    "preprocessing_params",
    "event_selection_rules",
    "pair_construction",
    "tic_partition_assignments",
    "random_seed",
    "model_checkpoint",
    "thresholds",
    "injection_params",
    "holdout_unblinding_date",
)

# --- Negative-pair reporting (PRD User Story 9) ---
NEGATIVE_CATEGORIES: tuple[str, ...] = (
    "random",
    "morphology-matched",
    "same-TIC hard",
)


@dataclass(frozen=True)
class Protocol:
    version: str = PROTOCOL_VERSION
    primary_product: str = PRIMARY_PHOTOMETRIC_PRODUCT
    dev_sectors: FrozenSet[int] = field(default=DEV_SECTORS)
    sealed_sectors: FrozenSet[int] = field(default=SEALED_SECTORS)
    discovery_sectors: FrozenSet[int] = field(default=DISCOVERY_SECTORS)
    min_period_days: float = LONG_PERIOD_LOWER_BOUND_DAYS
    primary_metric: str = PRIMARY_METRIC


PROTOCOL = Protocol()


def is_dev_sector(sector: int) -> bool:
    return sector in DEV_SECTORS


def is_sealed_sector(sector: int) -> bool:
    return sector in SEALED_SECTORS


def validate_no_temporal_leak(dev_sectors_used: FrozenSet[int] | set[int]) -> None:
    """Raise if any sealed/discovery sector leaks into development inputs."""
    leak = set(dev_sectors_used) & set(SEALED_SECTORS | DISCOVERY_SECTORS)
    if leak:
        raise ValueError(f"temporal leak: sealed sectors in dev inputs: {sorted(leak)}")


def validate_tic_partition(*partitions: FrozenSet[int] | set[int]) -> None:
    """Raise if any TIC appears in more than one partition."""
    seen: set[int] = set()
    for part in partitions:
        overlap = seen & set(part)
        if overlap:
            raise ValueError(f"TIC partition leak: {sorted(overlap)[:5]}")
        seen |= set(part)
