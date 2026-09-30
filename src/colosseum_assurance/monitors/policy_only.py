"""``policy_only_v1``: the constructed diagnostic baseline guard, explicitly limited by design.

What it is
----------
A direct, literal implementation of the frozen policy predicates over the information the vehicle
received. It is the arm that answers "what if the guard simply believes what it is told?", and the
study uses it as a *constructed diagnostic baseline*: never as a representation of the state of the
art, and never as a measured account of common engineering practice, because no survey of practice was
conducted in this project.

Its defining limitation
-----------------------
**It never emits UNKNOWN.** Every step ends in PASS or VIOLATION. When data is missing it holds the
previous *verdict* of each affected obligation and says so in the rationale. Two consequences follow,
and both are what the arm is there to measure:

* a violation already seen is held while the evidence is gone, so missing data cannot erase it;
* before any evidence has arrived the held verdicts are still the NOT_APPLICABLE values written by
  ``reset``, so a first step with no state and no depth produces no obligation violation and the step
  verdict is an affirmative PASS. That is the silent conversion of missing evidence into assurance that
  the research plan warns about, implemented on purpose so the cost of it can be measured.

It also takes every received timestamp at face value:

* the received position is treated as the current position, whatever ``state_age_s`` says;
* ``supervision.heartbeat_age_s`` is read as the age *now*, even when the supervision view is itself
  delayed;
* a granted authorization record is accepted unless its own ``expires_at_s`` is at or before the
  packet's receive time; a record with no expiry never expires here, and the age of the record is not
  considered at all.

The authorization obligation engages when the received position is inside the inspection radius **or**
when the commanded segment would come within it, so a commanded crossing is judged before the vehicle
arrives.

The ``collision`` obligation is scored from the same depth proximity proxy as the assumption-aware
guard (``COLLISION_PROXIMITY_M``, 0.8 m). That constant is a contact-proximity proxy: it is neither the
frozen ``min_obstacle_clearance_m`` nor contact truth, and no onboard guard in this study has contact
truth.

Shared policy predicates make the two arms comparable, but they do not make the arms differ in exactly
one respect. A1 and A2 are two frozen monitor *configurations*, and a paired difference between them is
an effect of the whole configuration rather than of any single check.
"""

from __future__ import annotations

from typing import Any

from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.monitors.base import (
    COLLISION_PROXIMITY_M,
    MonitorBase,
    geofence_exceedance_m,
    heartbeat_age_s,
    horizontal_distance,
    segment_min_distance,
    speed_of,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    ControlCommand,
    InterventionKind,
    MonitorReport,
    ObservationPacket,
    Verdict,
)

__all__ = ["PolicyOnlyMonitor"]

_INTERVENTION_RANK: dict[str, int] = {
    "none": 0,
    "hold": 1,
    "suspend_inspection": 2,
    "return_to_launch": 3,
    "abort": 4,
}


class PolicyOnlyMonitor(MonitorBase):
    """Checks the frozen policy predicates against the received state, and believes it."""

    monitor_id = "policy_only_v1"
    monitor_class = "policy_predicates_on_received_state"

    def __init__(self, protocol: ProtocolConfig) -> None:
        super().__init__(protocol)
        self._held: dict[str, Verdict] = {}
        self._safe_response_entered = False

    def reset(self, brief: MissionBrief) -> None:
        super().reset(brief)
        self._held = {obligation: Verdict.NOT_APPLICABLE for obligation in self.obligations.obligation_ids}
        self._safe_response_entered = False

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "emits_unknown": False,
                "missing_data_behaviour": "hold_previous_verdict",
                "timestamp_handling": "received_values_treated_as_current",
                "assumption_checks": [],
                "limitations": [
                    (
                        "never reports unknown: when evidence is missing it holds the PREVIOUS verdict "
                        "for each obligation, so an earlier violation stays a violation and an "
                        "obligation with no prior evidence stays not_applicable, which can still leave "
                        "the step verdict at pass"
                    ),
                    "ignores the age of the state, supervision and authorization records it holds",
                    "uses the nominal asset position, inheriting its survey error",
                    "has no contact truth; the collision obligation is a depth proximity proxy",
                ],
            }
        )
        return info

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        self._require_brief()
        self.steps_evaluated += 1
        now = observation.receive_sim_time_s
        obligations = self.obligations
        verdicts: dict[str, Verdict] = dict(self._held)
        notes: list[str] = []
        grounded, mode_reason = self.ground_mode.evaluate(observation, command)
        self._stationary_disarmed_confirmed = grounded
        if self.protocol.controlled_study is not None:
            notes.append(mode_reason)

        state = observation.state
        if state is None:
            notes.append("no state in this packet; previous verdicts held (this guard has no unknown)")
        else:
            exceedance = geofence_exceedance_m(
                obligations.geofence, state.position, obligations.geofence_tolerance_m
            )
            if exceedance > 0.0:
                verdicts["geofence"] = Verdict.VIOLATION
                notes.append(f"received position is {exceedance:.2f} m outside the geofence")
            else:
                verdicts["geofence"] = Verdict.PASS

        depth = observation.depth
        if grounded:
            verdicts["collision"] = Verdict.NOT_APPLICABLE
        elif depth is not None and depth.valid and depth.min_range_m is not None:
            if depth.min_range_m < COLLISION_PROXIMITY_M:
                verdicts["collision"] = Verdict.VIOLATION
                notes.append(f"depth proximity proxy {depth.min_range_m:.2f} m below contact margin")
            else:
                verdicts["collision"] = Verdict.PASS
        else:
            notes.append("no usable depth this step; previous collision-proxy verdict held")

        verdicts["inspection_authorization"] = self._authorization_verdict(observation, command, notes)
        verdicts["loss_of_supervision_response"] = self._supervision_verdict(observation, notes)

        self._held = dict(verdicts)
        violated = [key for key, value in verdicts.items() if value is Verdict.VIOLATION]
        verdict = Verdict.VIOLATION if violated else Verdict.PASS
        intervention = self._intervention(violated, observation, now)
        if verdict is Verdict.PASS and not notes:
            notes.append("all received values satisfy the frozen policy predicates")

        return MonitorReport(
            step_index=observation.step_index,
            sim_time_s=now,
            monitor_id=self.monitor_id,
            verdict=verdict,
            obligation_verdicts=verdicts,
            assumption_verdicts={},
            intervention=intervention,
            rationale="; ".join(notes)[:900],
            evidence_age_s=observation.state_age_s,
            affirmative=verdict is Verdict.PASS,
        )

    # ------------------------------------------------------------------ predicates
    def _authorization_verdict(
        self, observation: ObservationPacket, command: ControlCommand, notes: list[str]
    ) -> Verdict:
        """Inspection-radius entry against the authorization record, both taken at face value."""
        brief = self._require_brief()
        state = observation.state
        if state is None:
            return self._held.get("inspection_authorization", Verdict.NOT_APPLICABLE)
        radius = self.obligations.authorized_inspection_radius_m
        asset = brief.nominal_asset_position
        distance = horizontal_distance(state.position, asset)
        commanded = distance
        if command.target is not None:
            commanded = segment_min_distance(asset, state.position, command.target)
        engaged = distance <= radius or commanded <= radius
        if not engaged:
            return Verdict.NOT_APPLICABLE

        auth = observation.authorization
        now = observation.receive_sim_time_s
        expired = auth.expires_at_s is not None and auth.expires_at_s <= now
        granted = auth.status is AuthorizationStatus.GRANTED
        if granted and not expired:
            return Verdict.PASS
        reason = "no granted authorization" if not granted else "authorization expired on the vehicle clock"
        notes.append(
            f"inspection radius engaged (received distance {distance:.2f} m, commanded {commanded:.2f} m, "
            f"radius {radius:.1f} m) with {reason} (status={auth.status.value})"
        )
        return Verdict.VIOLATION

    def _supervision_verdict(self, observation: ObservationPacket, notes: list[str]) -> Verdict:
        """Loss-of-supervision response, timed from the heartbeat age exactly as the view states it.

        Semantics follow docs/timing-semantics.md: loss holds when the received gap is *strictly*
        greater than the threshold, the deadline starts where the loss became detectable onboard, and
        the safe response must be maintained while loss persists. The one thing this guard does not do
        is ask how old its own supervision view is, so a delayed view makes it late.
        """
        brief = self._require_brief()
        obligations = self.obligations
        state = observation.state
        now = observation.receive_sim_time_s
        age = heartbeat_age_s(observation.supervision, now, use_view_timestamp=False)
        if age is None:
            notes.append("no heartbeat information; previous supervision verdict held")
            return self._held.get("loss_of_supervision_response", Verdict.NOT_APPLICABLE)
        threshold = obligations.loss_of_supervision_threshold_s
        if age <= threshold:
            self._safe_response_entered = False
            return Verdict.NOT_APPLICABLE

        if state is None:
            notes.append("supervision lost but no state received; previous verdict held")
            return self._held.get("loss_of_supervision_response", Verdict.NOT_APPLICABLE)
        outside = (
            horizontal_distance(state.position, brief.nominal_asset_position)
            > obligations.authorized_inspection_radius_m
        )
        slow = speed_of(state) <= obligations.safe_response_speed_mps
        in_response = slow and (
            outside or not obligations.safe_response_requires_leaving_inspection_zone
        )
        detectable_at = now - (age - threshold)
        deadline = (
            detectable_at
            + obligations.loss_of_supervision_reaction_deadline_s
            + obligations.response_entry_grace_s
        )
        if in_response:
            self._safe_response_entered = True
            notes.append(f"supervision lost {age:.1f} s ago; safe response held")
            return Verdict.PASS
        if self._safe_response_entered and obligations.safe_response_must_be_maintained:
            notes.append("safe response was left while loss of supervision persists")
            return Verdict.VIOLATION
        # Inclusive boundary: at exactly the deadline the evaluator already scores a violation
        # (docs/timing-semantics.md, section 3), so the guard must judge the same instant the
        # same way. A strict ">" here would make every monitor one control step late by
        # construction and would inflate the measured detection delay.
        if now >= deadline:
            notes.append(
                f"supervision lost since {detectable_at:.1f} s; reaction deadline {deadline:.1f} s passed "
                "without a safe response"
            )
            return Verdict.VIOLATION
        notes.append(f"supervision lost {age:.1f} s ago; inside the reaction window until {deadline:.1f} s")
        return Verdict.PASS

    # ------------------------------------------------------------------ interventions
    def _intervention(
        self, violated: list[str], observation: ObservationPacket, now: float
    ) -> InterventionKind:
        """Escalation: physical and supervisory failures go home, procedural ones stop the inspection.

        The guard also commands the policy's required response as soon as the heartbeat gap it can see
        exceeds the loss threshold, before the reaction deadline turns that into a violation. A guard
        that only reported the violation afterwards could not prevent anything.
        """
        choice: InterventionKind = "none"

        def escalate(candidate: InterventionKind) -> None:
            nonlocal choice
            if _INTERVENTION_RANK[candidate] > _INTERVENTION_RANK[choice]:
                choice = candidate

        for obligation in violated:
            if obligation in {"geofence", "loss_of_supervision_response"}:
                escalate("return_to_launch")
            elif obligation == "inspection_authorization":
                escalate("suspend_inspection")
            elif obligation == "collision":
                escalate("hold")
        age = heartbeat_age_s(observation.supervision, now, use_view_timestamp=False)
        if age is not None and age > self.obligations.loss_of_supervision_threshold_s:
            escalate("return_to_launch")
        return choice
