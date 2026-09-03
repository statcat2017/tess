"""Injection operating-regime study (issue #7).

Thin facade: geometry, single-cell pipeline, grid/reporting, and learned
comparison live in dedicated modules. This module re-exports the public
contract so existing imports keep working.

Labels stay principled: cross-sector same-shape injected pairs are
positives; same-sector pairs are timing negatives and cross-sector
shape-mismatched pairs are morphology negatives. Raw archive data and
the sealed holdout are never touched.
"""

from __future__ import annotations

from tess_assoc.inject_cell import (
    best_distinct_target,
    candidate_learn_payload,
    cell_group_key,
    failure_mode,
    inject_transit,
    study_cell,
)
from tess_assoc.inject_geometry import (
    DEPTHS,
    SAME_EPOCH_DT_DAYS,
    SHAPES,
    feasible_interval,
    feasible_pair,
    supported_pair,
)
from tess_assoc.inject_grid import run_grid, summarize_grid
from tess_assoc.inject_learn import learned_comparison

__all__ = [
    "DEPTHS",
    "SHAPES",
    "SAME_EPOCH_DT_DAYS",
    "best_distinct_target",
    "candidate_learn_payload",
    "cell_group_key",
    "failure_mode",
    "feasible_interval",
    "feasible_pair",
    "inject_transit",
    "learned_comparison",
    "run_grid",
    "study_cell",
    "summarize_grid",
    "supported_pair",
]
