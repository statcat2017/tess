"""TESS multi-epoch transit association package (Phase 0: protocol freeze)."""

from tess_assoc.benchmark import (
    assign_partitions,
    build_benchmark,
    render_benchmark_report,
)
from tess_assoc.event import EventRecord
from tess_assoc.discovery import (
    DISCOVERY_SECTOR,
    cone_tics,
    load_discovery_manifest,
    render_discovery_report,
    run_discovery,
    sectors_for_tic,
    select_cohort,
)
from tess_assoc.freeze import (
    FreezeRecord,
    audit_development,
    check_manifest_bytes,
    create_freeze,
    load_holdout_manifest,
    mark_unblinded,
    verify_freeze,
)
from tess_assoc.holdout import (
    render_holdout_report,
    run_holdout,
)
from tess_assoc.inject import (
    inject_transit,
    learned_comparison,
    run_grid,
    study_cell,
    summarize_grid,
)
from tess_assoc.learn import run_comparison
from tess_assoc.orbit import generate_aliases, predict_epochs
from tess_assoc.pipeline import render_report, run_records, run_tracer
from tess_assoc.protocol import (
    SectorRole,
    validate_no_temporal_leak,
    validate_tic_partition,
)
from tess_assoc.vetting import (
    check_contamination,
    cross_match_toi,
    promote_candidate,
    secondary_search,
)
from tess_assoc.replay import load_replay_manifest, replay_all

__all__ = [
    "EventRecord",
    "DISCOVERY_SECTOR",
    "FreezeRecord",
    "SectorRole",
    "assign_partitions",
    "audit_development",
    "build_benchmark",
    "check_contamination",
    "check_manifest_bytes",
    "cone_tics",
    "create_freeze",
    "cross_match_toi",
    "generate_aliases",
    "inject_transit",
    "learned_comparison",
    "load_discovery_manifest",
    "load_holdout_manifest",
    "load_replay_manifest",
    "mark_unblinded",
    "predict_epochs",
    "promote_candidate",
    "render_benchmark_report",
    "render_discovery_report",
    "render_holdout_report",
    "render_report",
    "replay_all",
    "run_comparison",
    "run_discovery",
    "run_grid",
    "run_holdout",
    "run_records",
    "run_tracer",
    "secondary_search",
    "sectors_for_tic",
    "select_cohort",
    "study_cell",
    "summarize_grid",
    "validate_no_temporal_leak",
    "validate_tic_partition",
    "verify_freeze",
]
