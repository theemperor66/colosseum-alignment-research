"""Shared, verdict-free plumbing for the two runtime guards.

Deliberately small. The two guards must be able to disagree, so nothing here decides anything: this
module holds the abstract lifecycle, pure geometry and time arithmetic, and the thresholds that both
guards must share for the comparison to be fair. Every verdict function lives in
:mod:`colosseum_assurance.monitors.policy_only` or
:mod:`colosseum_assurance.monitors.assumption_aware`, written out separately in each, so that one
mistake cannot make both arms agree for the same wrong reason.

Guards see exactly what the controller sees: an :class:`~colosseum_assurance.schemas.ObservationPacket`
and a :class:`~colosseum_assurance.interfaces.MissionBrief`. In particular a guard knows only the
*nominal* asset position, so its inspection-radius test inherits the survey error of that nominal
position. That is a real evidence limitation of onboard monitoring and is reported, not hidden.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationView,
    Box,
    ControlCommand,
    MonitorReport,
    ObservationPacket,
    SupervisionView,
    Vec3,
    VehicleState,
)

__all__ = [
    "ASSET_POSITION_UNCERTAINTY_M",
    "COLLISION_PROXIMITY_M",
    "MonitorBase",
    "authorization_remaining_s",
    "geofence_exceedance_m",
    "heartbeat_age_s",
    "horizontal_distance",
    "received_heartbeat_time_s",
    "reachable_radius_m",
    "segment_min_distance",
    "speed_of",
]

COLLISION_PROXIMITY_M = 0.8
"""Depth range below which a return counts as an imminent-contact proxy.

Onboard monitors have no collision truth; the independent evaluator does. Both guards therefore score
the ``collision`` obligation from the same depth proximity proxy, and the proxy is labelled as such in
every rationale so that no reader mistakes it for a contact measurement.
"""

ASSET_POSITION_UNCERTAINTY_M = 2.0
"""Declared survey uncertainty of the nominal asset position, in metres.

The scenario generator displaces the asset from nominal by up to 1.0 m in x and 1.5 m in y. Declaring
that bound is an assumption statement, like the declared observation delay; it is not the per-scenario
truth, which no guard may read. Guards that reason about assumptions may use it as a margin.
"""


# ----------------------------------------------------------------------------------------------
# Pure geometry and time arithmetic (no verdicts)
# ----------------------------------------------------------------------------------------------
def horizontal_distance(a: Vec3, b: Vec3) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def geofence_exceedance_m(box: Box, point: Vec3, tolerance_m: float) -> float:
    """Metres by which ``point`` lies outside ``box`` beyond ``tolerance_m``; <= 0 means inside."""
    return box.signed_exceedance(point) - tolerance_m


def segment_min_distance(point: Vec3, start: Vec3, end: Vec3) -> float:
    """Closest horizontal approach of the segment ``start``-``end`` to ``point``.

    Used to ask whether a commanded motion would enter a radius, not only whether the vehicle is in it.
    """
    px, py = point.x - start.x, point.y - start.y
    dx, dy = end.x - start.x, end.y - start.y
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * dx + py * dy) / denom))
    return math.hypot(px - t * dx, py - t * dy)


def reachable_radius_m(max_speed_mps: float, elapsed_s: float, extra_m: float = 0.0) -> float:
    """Radius the vehicle could have covered in ``elapsed_s`` at its bounded speed, plus a margin."""
    return max(0.0, max_speed_mps) * max(0.0, elapsed_s) + max(0.0, extra_m)


def speed_of(state: VehicleState) -> float:
    return math.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)


def received_heartbeat_time_s(supervision: SupervisionView) -> float | None:
    """Receipt time of the newest heartbeat the vehicle holds, ``r(t)`` in docs/timing-semantics.md.

    Receipt, not production, is the operative time: the vehicle cannot act on a message it never got.
    """
    if supervision.last_heartbeat_received_at_s is not None:
        return supervision.last_heartbeat_received_at_s
    if supervision.heartbeat_age_s is not None:
        return supervision.sim_time_s - supervision.heartbeat_age_s
    if supervision.last_heartbeat_sim_time_s is not None:
        return supervision.last_heartbeat_sim_time_s
    return None


def heartbeat_age_s(supervision: SupervisionView, now_s: float, *, use_view_timestamp: bool) -> float | None:
    """Age of the newest received heartbeat.

    ``use_view_timestamp=False`` returns the age exactly as the view states it, measured on the view's
    own clock. That is what a guard which treats received information as current would use.
    ``use_view_timestamp=True`` re-times the receipt against ``now_s``, which is what a guard that
    checks the age of its own evidence would use. The two differ whenever the supervision view is itself
    delayed, and that difference is one of the effects under study.
    """
    if not use_view_timestamp:
        if supervision.heartbeat_age_s is not None:
            return supervision.heartbeat_age_s
        received = received_heartbeat_time_s(supervision)
        return None if received is None else supervision.sim_time_s - received
    received = received_heartbeat_time_s(supervision)
    return None if received is None else max(0.0, now_s - received)


def authorization_remaining_s(authorization: AuthorizationView, now_s: float) -> float | None:
    """Seconds of validity left at ``now_s``; ``None`` when the record carries no expiry."""
    if authorization.expires_at_s is None:
        return None
    return authorization.expires_at_s - now_s


# ----------------------------------------------------------------------------------------------
# Lifecycle base
# ----------------------------------------------------------------------------------------------
class MonitorBase(ABC):
    """Lifecycle and bookkeeping shared by the guards. Produces no verdicts."""

    monitor_id: ClassVar[str] = "abstract_monitor"
    monitor_class: ClassVar[str] = "abstract"

    def __init__(self, protocol: ProtocolConfig) -> None:
        from colosseum_assurance.monitors.ground_mode import GroundModeTracker

        self.protocol = protocol
        self.mission = protocol.mission
        self.obligations = protocol.obligations
        self.brief: MissionBrief | None = None
        self.steps_evaluated = 0
        self.ground_mode = GroundModeTracker(protocol)
        self._stationary_disarmed_confirmed = False

    def reset(self, brief: MissionBrief) -> None:
        """Start a new episode. Subclasses extend this to clear their own memory."""
        self.brief = brief
        self.mission = brief.mission
        self.obligations = brief.obligations
        self.steps_evaluated = 0
        from colosseum_assurance.monitors.ground_mode import GroundModeTracker

        self.ground_mode = GroundModeTracker(self.protocol)
        self._stationary_disarmed_confirmed = False

    @abstractmethod
    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport: ...

    def describe(self) -> dict[str, Any]:
        """Static self-description written into the evidence record and the comparator documentation."""
        return {
            "monitor_id": self.monitor_id,
            "monitor_class": self.monitor_class,
            "policy_version": self.obligations.policy_version,
            "obligation_ids": list(self.obligations.obligation_ids),
            "collision_proximity_m": COLLISION_PROXIMITY_M,
            "asset_position_reference": "mission_brief_nominal_asset_position",
        }

    # ------------------------------------------------------------------ small shared context
    def _require_brief(self) -> MissionBrief:
        if self.brief is None:
            raise RuntimeError(f"{self.monitor_id}.evaluate called before reset(brief)")
        return self.brief

    def _declared_observation_delay_s(self, observation: ObservationPacket) -> float:
        """Declared bound, from the packet when present, otherwise from the brief. Never measured."""
        brief = self._require_brief()
        if observation.declared_observation_delay_s is not None:
            return observation.declared_observation_delay_s
        return brief.declared_observation_delay_s
