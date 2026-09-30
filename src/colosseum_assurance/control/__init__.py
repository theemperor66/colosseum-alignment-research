"""Vehicle-side code: perception and the fixed goal-directed controller.

Nothing in this package may import scenario geometry, privileged truth, or evaluation code. The
leakage test ``tests/unit/test_no_privileged_leakage.py`` enforces that mechanically.
"""

from __future__ import annotations

from colosseum_assurance.control.controller import InspectionController
from colosseum_assurance.control.perception import (
    PerceptionParams,
    TargetDetection,
    summarize_depth,
)

__all__ = ["InspectionController", "PerceptionParams", "TargetDetection", "summarize_depth"]
