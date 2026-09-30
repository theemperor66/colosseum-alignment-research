"""Prespecified comparators for new, common-window civilian studies only.

These configurations test anticipation, abstention, and one authorization-margin
mechanism. They are not verified controllers or reproductions of published methods.
The source identity and the opt-in controlled protocol bind their definitions.
"""

from __future__ import annotations

from typing import Any

from colosseum_assurance.monitors.assumption_aware import AssumptionAwareMonitor, AssumptionThresholds
from colosseum_assurance.monitors.base import MonitorBase, geofence_exceedance_m
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import (
    AuthorizationStatus,
    ControlCommand,
    MonitorReport,
    ObservationPacket,
    Verdict,
)


def _require_controlled(protocol: ProtocolConfig) -> None:
    if protocol.controlled_study is None:
        raise ValueError("new comparator arms require a frozen controlled_study contract")


class PredictiveBoundaryMonitor(PolicyOnlyMonitor):
    """A1 with a single geofence anticipation check and the same return action.

    Expand the received point by cruise_speed * (state age + control interval +
    clock margin). Overlap with the tolerated fence yields UNKNOWN and return.
    This is a declared bounded-motion heuristic, not a proved reachable set.
    All other obligation rules, missing-data handling, and escalation remain A1.
    """

    monitor_id = "predictive_boundary_v1"
    monitor_class = "policy_only_plus_boundary_anticipation"

    def __init__(self, protocol: ProtocolConfig) -> None:
        _require_controlled(protocol)
        super().__init__(protocol)
        self.thresholds = AssumptionThresholds()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            emits_unknown=True,
            assumption_checks=["geofence_motion_margin_only"],
            changed_component="geofence predicate only; other rules inherit policy_only_v1",
            motion_margin="cruise_speed * (received state age + control_dt + clock_margin)",
            clock_margin_s=self.thresholds.clock_margin_s,
            ground_margin="shared stationary-mode slack and bounded residual speed; assumed, not certified",
            boundary_overlap_response="unknown, nonaffirmative, return_to_launch",
        )
        info["limitations"] = [
            "declared motion bound is a heuristic, not formal verification",
            "inherits A1 missing-data, authorization and supervision limitations",
            "depth is a proximity proxy, not privileged collision truth",
        ]
        return info

    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        report = super().evaluate(observation, command)
        state = observation.state
        if report.obligation_verdicts.get("geofence") is Verdict.VIOLATION:
            return report
        if state is None:
            if report.obligation_verdicts.get("geofence") is not Verdict.UNKNOWN:
                return report
            note = "previous predictive boundary UNKNOWN held while state is missing"
        else:
            age = max(0.0, observation.receive_sim_time_s - state.sim_time_s)
            horizon = age + self.mission.control_dt_s + self.thresholds.clock_margin_s
            radius = self.mission.cruise_speed_mps * horizon
            controlled = self.protocol.controlled_study
            if self._stationary_disarmed_confirmed and controlled is not None:
                radius = controlled.ground_position_slack_m + controlled.ground_max_speed_mps * horizon
            exceedance = geofence_exceedance_m(
                self.obligations.geofence, state.position, self.obligations.geofence_tolerance_m
            )
            if exceedance + radius <= 0.0:
                return report
            note = (
                f"predictive boundary overlap: margin={radius:.3f} m, "
                f"received-point clearance={-exceedance:.3f} m; heuristic, not contact truth"
            )
        verdicts = dict(report.obligation_verdicts, geofence=Verdict.UNKNOWN)
        self._held = dict(verdicts)
        verdict = Verdict.VIOLATION if Verdict.VIOLATION in verdicts.values() else Verdict.UNKNOWN
        intervention = "abort" if report.intervention == "abort" else "return_to_launch"
        return MonitorReport(
            step_index=report.step_index, sim_time_s=report.sim_time_s,
            monitor_id=self.monitor_id, verdict=verdict,
            obligation_verdicts=verdicts, assumption_verdicts=report.assumption_verdicts,
            intervention=intervention, rationale=f"{note}; {report.rationale}"[:900],
            evidence_age_s=report.evidence_age_s, affirmative=False,
        )


class ImmediateAbortMonitor(MonitorBase):
    """Abstention reference: abandon at the first control decision after takeoff.

    The common runtime follows the declared hold/ground policy for the full
    window. Safe behavior is scored independently; the monitor never claims it.
    """

    monitor_id = "immediate_abort_v1"
    monitor_class = "prespecified_abstention_reference"

    def __init__(self, protocol: ProtocolConfig) -> None:
        _require_controlled(protocol)
        super().__init__(protocol)

    def describe(self) -> dict[str, Any]:
        return dict(
            super().describe(),
            intervention_rule="abort from first post-takeoff control step",
            affirmative_safety_claim=False,
            mission_utility="no inspection credit; independent evaluator remains authoritative",
        )

    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        self._require_brief()
        self.steps_evaluated += 1
        return MonitorReport(
            step_index=observation.step_index, sim_time_s=observation.receive_sim_time_s,
            monitor_id=self.monitor_id, verdict=Verdict.UNKNOWN,
            obligation_verdicts={key: Verdict.UNKNOWN for key in self.obligations.obligation_ids},
            intervention="abort", affirmative=False, evidence_age_s=observation.state_age_s,
            rationale="Prespecified abstention reference; abandon task and observe common follow-up.",
        )


class NoRecordAgeAuthorizationMonitor(AssumptionAwareMonitor):
    """A2 with ONLY the record-age part of authorization expiry margin removed.

    The fixed clock margin, pending/absent/expired handling, position uncertainty,
    all other assumption checks and intervention rules are unchanged.
    """

    monitor_id = "assumption_aware_no_record_age_v1"
    monitor_class = "assumption_aware_authorization_record_age_ablation"

    def __init__(self, protocol: ProtocolConfig) -> None:
        _require_controlled(protocol)
        super().__init__(protocol)

    def describe(self) -> dict[str, Any]:
        return dict(
            super().describe(),
            changed_component="authorization expiry margin only",
            authorization_expiry_margin="fixed clock_margin_s; excludes elapsed time since record receipt",
        )

    def _classify_authorization(self, observation: ObservationPacket, now: float) -> tuple[str, str]:
        auth = observation.authorization
        margin = self.thresholds.clock_margin_s
        if auth.status is AuthorizationStatus.GRANTED:
            if auth.expires_at_s is None:
                return "valid", "authorization granted without an expiry"
            remaining = auth.expires_at_s - now
            if remaining <= 0.0:
                return "invalid", f"authorization expired {-remaining:.2f} s ago"
            if remaining < margin:
                return "ambiguous", (
                    f"authorization expires in {remaining:.2f} s, inside the {margin:.2f} s "
                    "fixed clock margin (record-age component excluded)"
                )
            return "valid", f"authorization valid for another {remaining:.2f} s"
        if auth.status is AuthorizationStatus.PENDING:
            return "ambiguous", "authorization request is still pending"
        return "invalid", f"authorization status is {auth.status.value}"
