"""The confidence guard consumes onboard predictions only and cannot erase base safety responses."""

from __future__ import annotations

import pytest
from tests.unit.test_monitor_policy_only import FAR_FROM_ASSET, make_brief, make_packet

from colosseum_assurance.monitors.perception_guard import PerceptionConfidenceGuard
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import ControlCommand, MonitorReport, Verdict


def packet(**updates):
    observation = make_packet(4, 10.0, FAR_FROM_ASSET, heartbeat_received_at_s=10.0)
    # Explicit test predictions, not trained-model or simulator results.
    return observation.model_copy(update={
        "asset_presence_probability": 0.85, "perception_model_hash": "sha256:" + "a" * 64,
        "perception_prediction_time_s": 9.5, **updates,
    })


def command(kind="inspect_capture"):
    return ControlCommand(step_index=4, issued_sim_time_s=10.0, kind=kind)


def guard():
    result = PerceptionConfidenceGuard(PolicyOnlyMonitor(ProtocolConfig()), 0.8, max_age_s=1.0)
    result.reset(make_brief())
    return result


@pytest.mark.parametrize("updates", [
    {"asset_presence_probability": None}, {"asset_presence_probability": 0.79},
    {"asset_presence_probability": float("nan")}, {"asset_presence_probability": -0.1},
    {"perception_model_hash": None}, {"perception_model_hash": "unanchored"},
    {"perception_model_hash": "a" * 64},
    {"perception_prediction_time_s": None}, {"perception_prediction_time_s": 10.1},
    {"perception_prediction_time_s": 8.999},
])
def test_missing_low_invalid_or_stale_prediction_is_unknown_and_holds(updates):
    report = guard().evaluate(packet(**updates), command())
    assert report.verdict is Verdict.UNKNOWN
    assert report.assumption_verdicts["perception_capture_confidence"] is Verdict.UNKNOWN
    assert not report.affirmative and report.intervention == "hold"
    assert "capture confidence undecidable" in report.rationale
    assert Verdict.VIOLATION not in report.obligation_verdicts.values()


def test_threshold_and_age_boundaries_are_inclusive_without_changing_base_obligations():
    monitor = guard()
    report = monitor.evaluate(packet(asset_presence_probability=0.8,
                                     perception_prediction_time_s=9.0), command())
    assert report.verdict is Verdict.PASS and report.affirmative
    assert report.intervention == "none"
    assert report.assumption_verdicts["perception_capture_confidence"] is Verdict.PASS
    assert report.monitor_id == monitor.monitor_id


def test_fitted_model_probability_and_canonical_hash_reach_capture_guard():
    from tests.unit.test_perception_eval import fixture_rows

    from colosseum_assurance.perception_eval import fit_model, predict_probability

    rows = fixture_rows()
    model = fit_model(rows)
    # Train/calibrate on separate synthetic scenario groups, then consume a held-out prediction.
    positive = next(r for r in rows if r.split == "test" and r.label.value == 1)
    negative = next(r for r in rows if r.split == "test" and r.label.value == 0)
    assert model.model_hash.startswith("sha256:")
    monitor = guard()
    high = monitor.evaluate(packet(
        asset_presence_probability=predict_probability(model, positive.features),
        perception_model_hash=model.model_hash), command())
    low = monitor.evaluate(packet(
        asset_presence_probability=predict_probability(model, negative.features),
        perception_model_hash=model.model_hash), command())
    assert high.verdict is Verdict.PASS and high.intervention == "none"
    assert low.verdict is Verdict.UNKNOWN and low.intervention == "hold"


def test_gate_does_not_block_unrelated_commands_or_read_labels():
    no_prediction = packet(asset_presence_probability=None, perception_model_hash=None,
                           perception_prediction_time_s=None,
                           sensor_samples={"truth": {"asset_present": True}})
    changed_label = no_prediction.model_copy(update={
        "sensor_samples": {"truth": {"asset_present": False}}})
    first = guard().evaluate(no_prediction, command("move_to"))
    second = guard().evaluate(changed_label, command("move_to"))
    assert first == second
    assert first.verdict is Verdict.PASS and first.intervention == "none"
    assert first.assumption_verdicts["perception_capture_confidence"] is Verdict.NOT_APPLICABLE
    capture_one = guard().evaluate(no_prediction, command())
    capture_two = guard().evaluate(changed_label, command())
    assert capture_one == capture_two and capture_one.verdict is Verdict.UNKNOWN


@pytest.mark.parametrize("intervention", ["hold", "suspend_inspection", "return_to_launch", "abort"])
def test_base_violation_and_stronger_intervention_are_never_downgraded(intervention):
    class BaseResponse:
        monitor_id = "test_response"

        def reset(self, brief):
            self.brief = brief

        def describe(self):
            return {"monitor_id": self.monitor_id}

        def evaluate(self, observation, proposed):
            self.last_report = MonitorReport(
                step_index=observation.step_index, sim_time_s=observation.receive_sim_time_s,
                monitor_id=self.monitor_id, verdict=Verdict.VIOLATION,
                obligation_verdicts={"geofence": Verdict.VIOLATION},
                intervention=intervention, rationale="base geofence response")
            return self.last_report

    base = BaseResponse()
    monitor = PerceptionConfidenceGuard(base, 0.8)
    brief = make_brief()
    monitor.reset(brief)
    assert base.brief is brief
    report = monitor.evaluate(packet(asset_presence_probability=0.2), command())
    assert report.verdict is Verdict.VIOLATION and report.intervention == intervention
    assert report.obligation_verdicts == {"geofence": Verdict.VIOLATION}
    assert report.assumption_verdicts["perception_capture_confidence"] is Verdict.UNKNOWN
    assert base.last_report.assumption_verdicts == {}


@pytest.mark.parametrize("threshold,max_age", [(-0.01, 1), (1.01, 1), (float("nan"), 1),
                                             (True, 1), (0.8, -1), (0.8, float("inf"))])
def test_invalid_frozen_configuration_is_rejected(threshold, max_age):
    with pytest.raises(ValueError):
        PerceptionConfidenceGuard(PolicyOnlyMonitor(ProtocolConfig()), threshold, max_age)


def test_description_keeps_base_identity_and_states_narrow_capture_scope():
    description = guard().describe()
    assert description["base_monitor_id"] == "policy_only_v1"
    assert description["perception_guard"]["threshold"] == 0.8
    assert "inspect_capture only" in description["perception_guard"]["scope"]
    assert "does not establish flight safety" in description["perception_guard"]["limitation"]
