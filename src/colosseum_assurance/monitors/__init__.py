"""Runtime guards (arms A1 and A2) and the factory the runner uses to build them.

All guards see only onboard information. A1 and A2 are bundled configurations,
not a single-mechanism contrast. Opt-in controlled studies additionally include
anticipation, abstention, and a one-component authorization ablation.
"""

from __future__ import annotations

from colosseum_assurance.interfaces import Monitor
from colosseum_assurance.monitors.assumption_aware import AssumptionAwareMonitor
from colosseum_assurance.monitors.base import MonitorBase
from colosseum_assurance.monitors.controlled_comparators import (
    ImmediateAbortMonitor,
    NoRecordAgeAuthorizationMonitor,
    PredictiveBoundaryMonitor,
)
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.protocol.spec import ProtocolConfig

__all__ = ["AssumptionAwareMonitor", "MonitorBase", "PolicyOnlyMonitor", "build_monitor", "MONITOR_IDS"]

_MONITORS = (
    PolicyOnlyMonitor, AssumptionAwareMonitor, PredictiveBoundaryMonitor,
    ImmediateAbortMonitor, NoRecordAgeAuthorizationMonitor,
)
MONITOR_IDS: tuple[str, ...] = tuple(cls.monitor_id for cls in _MONITORS)


def build_monitor(monitor_id: str, protocol: ProtocolConfig) -> Monitor:
    """Build the guard named in an arm specification.

    Raises ``KeyError`` for an unknown id: an arm that silently ran without its guard would be
    unrecoverable evidence, so the failure has to be loud.
    """
    for cls in _MONITORS:
        if monitor_id == cls.monitor_id:
            return cls(protocol)
    raise KeyError(f"unknown monitor_id {monitor_id!r}; known monitors: {list(MONITOR_IDS)}")
