"""V4 capture permission from a bound, delayed observation; never a safety probability."""

from __future__ import annotations

import math
from typing import Any

from colosseum_assurance.interfaces import MissionBrief, Monitor
from colosseum_assurance.perception_eval.contracts import content_hash, validate_features
from colosseum_assurance.protocol.context_confidence import ContextConfidenceSpec
from colosseum_assurance.schemas import ControlCommand, MonitorReport, ObservationPacket, Verdict


class ContextConfidenceGuard:
    def __init__(self, base: Monitor, policy: ContextConfidenceSpec):
        self.base = base
        self.policy = policy.model_copy(deep=True)
        self.monitor_id = base.monitor_id + "+" + self.policy.semantics_version

    def reset(self, brief: MissionBrief) -> None:
        self.base.reset(brief)

    def describe(self) -> dict[str, Any]:
        result = dict(
            self.base.describe(),
            monitor_id=self.monitor_id,
            base_monitor_id=self.base.monitor_id,
            context_confidence=self.policy.model_dump(mode="json"),
            context_policy_hash=self.policy.content_hash(),
            limitation=(
                "Quality-only permission shares model-health and observed-frame eligibility; "
                "it ignores eligible probability magnitude and does not certify "
                "target content or safety."
                if self.policy.semantics_version == "observable_quality_capture_gate_v1"
                else "Permission uses delivered observed RGB; it does not predict safety or "
                "guarantee content of the newer image saved by execution."
            ),
        )
        if self.policy.semantics_version == "record_only_capture_evidence_v1":
            result["limitation"] = (
                "Confidence evidence is recorded only. Eligibility, quality and probability never "
                "change the base monitor verdict, affirmative decision or intervention."
            )
        return result

    def _decision(self, observation: ObservationPacket) -> dict[str, Any]:
        policy = self.policy
        evidence = observation.perception_evidence
        rgb = observation.rgb
        decision = dict(
            policy_hash=policy.content_hash(),
            context="unknown",
            threshold=None,
            probability=observation.asset_presence_probability,
            status="unknown",
            reason="prediction/frame evidence missing",
            observation_frame_id=None,
            observed_rgb_sha256=None,
            acquisition_time_s=None,
            evidence_age_s=None,
            probability_scope=policy.probability_claim,
        )
        if policy.semantics_version == "observable_quality_capture_gate_v1":
            decision.update(
                policy_semantics=policy.semantics_version, eligible_probability_magnitude_used=False
            )
        if evidence is None or rgb is None:
            return decision
        decision.update(
            observation_frame_id=evidence.frame_id,
            observed_rgb_sha256=evidence.rgb_sha256,
            acquisition_time_s=evidence.image_time_s,
            evidence_age_s=observation.receive_sim_time_s - evidence.image_time_s,
        )
        if (
            evidence.model_hash != policy.expected_model_hash
            or observation.perception_model_hash != policy.expected_model_hash
            or evidence.label_spec_hash != policy.expected_label_spec_hash
            or evidence.feature_version != policy.feature_version
        ):
            decision["reason"] = "model/feature/label identity differs from frozen policy"
            return decision
        if (
            evidence.camera_name != policy.camera_name
            or rgb.camera_name != policy.camera_name
            or rgb.kind != "rgb"
            or not rgb.acquisition_time_known
            or rgb.frame_id != evidence.frame_id
            or not evidence.frame_id
            or rgb.content_sha256 != evidence.rgb_sha256
            or not _canonical_hash(evidence.rgb_sha256)
            or rgb.width != evidence.width
            or rgb.height != evidence.height
            or rgb.sim_time_s != evidence.image_time_s
            or observation.perception_prediction_time_s != evidence.image_time_s
        ):
            decision["reason"] = "prediction is not bound to the delivered RGB frame/camera/time"
            return decision
        age = decision["evidence_age_s"]
        if not math.isfinite(age) or age < 0 or age > policy.max_evidence_age_s:
            decision["reason"] = "observed prediction is stale or future-dated"
            return decision
        try:
            validate_features(evidence.features)
        except ValueError:
            decision["reason"] = "observation feature contract is invalid"
            return decision
        if (
            content_hash(evidence.features) != evidence.features_sha256
            or evidence.features["depth_available"] not in (0.0, 1.0)
            or bool(evidence.features["depth_available"]) != evidence.depth_available
            or evidence.features["depth_valid_fraction"] != evidence.depth_valid_fraction
            or evidence.features["rgb_std"] != evidence.rgb_std
        ):
            decision["reason"] = "same-frame context features differ from their recorded binding"
            return decision
        probability = observation.asset_presence_probability
        if (
            probability is None
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or probability != evidence.probability
        ):
            decision["reason"] = "prediction probability missing or mismatched"
            return decision
        clear = (
            evidence.depth_available
            and evidence.depth_valid_fraction >= policy.clear_min_depth_fraction
            and evidence.rgb_std >= policy.clear_min_rgb_std
        )
        context = "clear_observation" if clear else "degraded_observation"
        if policy.semantics_version == "record_only_capture_evidence_v1":
            decision.update(context=context, status="eligible", reason="shared evidence eligibility met")
            return decision
        if policy.semantics_version == "observable_quality_capture_gate_v1":
            decision.update(
                context=context,
                status="pass" if clear else "unknown",
                reason="observed quality meets the declared clear-quality predicate"
                if clear
                else "observed quality does not meet the declared clear-quality predicate",
            )
            return decision
        threshold = policy.clear_threshold if clear else policy.degraded_threshold
        assert threshold is not None  # probability semantics are validated by the policy contract
        decision.update(
            context=context,
            threshold=threshold,
            status="pass" if probability >= threshold else "unknown",
            reason="observed asset probability meets declared context threshold"
            if probability >= threshold
            else "observed asset probability below context threshold",
        )
        return decision

    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        report = self.base.evaluate(observation, command)
        if self.policy.semantics_version == "record_only_capture_evidence_v1":
            decision = self._decision(observation) if command.kind == "inspect_capture" else None
            if decision is not None:
                decision.update(
                    policy_semantics=self.policy.semantics_version,
                    shared_evidence_eligible=decision["status"] == "eligible",
                    eligibility_status=decision["status"],
                    status="not_enforced",
                    confidence_veto_enabled=False,
                )
            return report.model_copy(
                update=dict(monitor_id=self.monitor_id, context_confidence_evidence=decision)
            )
        assumptions = dict(report.assumption_verdicts)
        key = (
            "observable_capture_quality"
            if self.policy.semantics_version == "observable_quality_capture_gate_v1"
            else "context_asset_capture_confidence"
        )
        if command.kind != "inspect_capture":
            assumptions[key] = Verdict.NOT_APPLICABLE
            return report.model_copy(update=dict(monitor_id=self.monitor_id, assumption_verdicts=assumptions))
        decision = self._decision(observation)
        passed = decision["status"] == "pass"
        assumptions[key] = Verdict.PASS if passed else Verdict.UNKNOWN
        changes = dict(
            monitor_id=self.monitor_id, assumption_verdicts=assumptions, context_confidence_evidence=decision
        )
        if not passed:
            changes.update(
                verdict=Verdict.VIOLATION if report.verdict is Verdict.VIOLATION else Verdict.UNKNOWN,
                affirmative=False,
                intervention="hold" if report.intervention == "none" else report.intervention,
                rationale=(report.rationale + "; " + decision["reason"]).lstrip("; "),
            )
        return report.model_copy(update=changes)


def _canonical_hash(value):
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(c in "0123456789abcdef" for c in value[7:])
    )
