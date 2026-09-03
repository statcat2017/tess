"""TESS multi-epoch transit association package (Phase 0: protocol freeze)."""

from tess_assoc.event import EventRecord
from tess_assoc.orbit import generate_aliases, predict_epochs
from tess_assoc.protocol import (
    SectorRole,
    validate_no_temporal_leak,
    validate_tic_partition,
)

__all__ = [
    "EventRecord",
    "SectorRole",
    "generate_aliases",
    "predict_epochs",
    "validate_no_temporal_leak",
    "validate_tic_partition",
]
