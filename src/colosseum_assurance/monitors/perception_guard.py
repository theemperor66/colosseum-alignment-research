"""Optional civilian capture-confidence gate over a frozen RGB predictor.

The gate uses only the delayed onboard prediction, its model hash and acquisition time. It never
reads segmentation labels, privileged asset poses or evaluator truth. A probability is evidence for
the capture decision, not a probability that the flight is safe.
"""

from __future__ import annotations

import math
from typing import Any

from colosseum_assurance.interfaces import MissionBrief, Monitor
from colosseum_assurance.schemas import ControlCommand, MonitorReport, ObservationPacket, Verdict


class PerceptionConfidenceGuard:
    """Keep the base guard's decisions; hold proposed captures when perception is undecidable."""

    def __init__(self, base: Monitor, threshold: float, max_age_s: float = 1.0) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("perception threshold must be finite and within [0, 1]")
        if isinstance(max_age_s, bool) or not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("perception maximum age must be finite and non-negative")
        self.base = base
        self.threshold = float(threshold)
        self.max_age_s = float(max_age_s)
        self.monitor_id = f"{base.monitor_id}+perception_confidence_v1"

    def reset(self, brief: MissionBrief) -> None:
        self.base.reset(brief)

    def describe(self) -> dict[str, Any]:
        return {
            **self.base.describe(), "monitor_id": self.monitor_id,
            "base_monitor_id": self.base.monitor_id,
            "perception_guard": {
                "threshold": self.threshold, "max_age_s": self.max_age_s,
                "scope": "proposed civilian inspect_capture only",
                "missing_or_low_confidence": "unknown; hold unless base intervention is stronger",
                "inputs": ["asset_presence_probability", "perception_model_hash",
                           "perception_prediction_time_s", "receive_sim_time_s"],
                "limitation": "Predictor confidence gates capture; it does not establish flight safety.",
            },
        }

    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        report = self.base.evaluate(observation, command)
        assumptions = dict(report.assumption_verdicts)
        changes: dict[str, Any] = {"monitor_id": self.monitor_id,
                                  "assumption_verdicts": assumptions}
        key = "perception_capture_confidence"
        if command.kind != "inspect_capture":
            assumptions[key] = Verdict.NOT_APPLICABLE
            return report.model_copy(update=changes)

        probability = observation.asset_presence_probability
        model_hash = observation.perception_model_hash
        prediction_time = observation.perception_prediction_time_s
        age = None if prediction_time is None else observation.receive_sim_time_s - prediction_time
        usable_model = (isinstance(model_hash, str) and model_hash.startswith("sha256:")
                        and len(model_hash) == 71 and all(
                            char in "0123456789abcdef" for char in model_hash[7:]))
        if not usable_model:
            reason = "prediction model hash unavailable or invalid"
        elif age is None or not math.isfinite(age) or age < 0 or age > self.max_age_s:
            reason = "prediction time unavailable, future-dated or stale"
        elif (probability is None or isinstance(probability, bool) or
              not math.isfinite(probability) or not 0 <= probability <= 1):
            reason = "asset-presence probability unavailable or invalid"
        elif probability < self.threshold:
            reason = (f"asset-presence probability {probability:.6g} below frozen threshold "
                      f"{self.threshold:.6g}")
        else:
            assumptions[key] = Verdict.PASS
            return report.model_copy(update=changes)

        assumptions[key] = Verdict.UNKNOWN
        changes.update(
            verdict=Verdict.VIOLATION if report.verdict is Verdict.VIOLATION else Verdict.UNKNOWN,
            affirmative=False,
            intervention="hold" if report.intervention == "none" else report.intervention,
            rationale=(report.rationale + "; capture confidence undecidable: " + reason).lstrip("; "),
        )
        return report.model_copy(update=changes)
