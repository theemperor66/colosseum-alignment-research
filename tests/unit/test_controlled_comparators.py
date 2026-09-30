"""Mechanism isolation, boundary cases and abstention accounting for new arms."""

import pytest
from tests.unit.test_monitor_policy_only import (
    granted_token,
    make_brief,
    make_command,
    make_packet,
)

from colosseum_assurance.monitors import build_monitor
from colosseum_assurance.monitors.assumption_aware import AssumptionAwareMonitor
from colosseum_assurance.monitors.controlled_comparators import (
    ImmediateAbortMonitor,
    NoRecordAgeAuthorizationMonitor,
    PredictiveBoundaryMonitor,
)
from colosseum_assurance.monitors.policy_only import PolicyOnlyMonitor
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.schemas import AuthorizationStatus, AuthorizationView, Verdict


def controlled_protocol():
    data = ProtocolConfig().model_dump(mode="json")
    data["protocol_schema_version"] = "3.0.0"
    data["controlled_study"] = {}
    return ProtocolConfig.model_validate(data)


def initialized(cls):
    monitor = cls(controlled_protocol())
    monitor.reset(make_brief())
    return monitor


@pytest.mark.parametrize("cls", [
    PredictiveBoundaryMonitor, ImmediateAbortMonitor, NoRecordAgeAuthorizationMonitor,
])
def test_new_comparators_require_opt_in_and_registered_factory(cls):
    with pytest.raises(ValueError, match="controlled_study"):
        build_monitor(cls.monitor_id, ProtocolConfig())
    assert isinstance(build_monitor(cls.monitor_id, controlled_protocol()), cls)


@pytest.mark.parametrize("x, expected", [(39.0, Verdict.PASS), (40.5, Verdict.UNKNOWN),
                                         (43.0, Verdict.VIOLATION)])
def test_prediction_precedes_violation_without_erasing_observed_violation(x, expected):
    monitor = initialized(PredictiveBoundaryMonitor)
    packet = make_packet(0, 1, (x, 0, -6))
    report = monitor.evaluate(packet, make_command(0, 1, kind="hold"))
    assert report.obligation_verdicts["geofence"] is expected
    if expected is Verdict.UNKNOWN:
        assert report.intervention == "return_to_launch"
        assert not report.affirmative
        assert report.verdict is Verdict.UNKNOWN


def test_prediction_keeps_all_non_geofence_policy_verdicts_identical():
    packet = make_packet(0, 10, (40.5, 0, -6), depth_min_range_m=.5, heartbeat_received_at_s=0)
    command = make_command(0, 10, kind="hold")
    base = initialized(PolicyOnlyMonitor).evaluate(packet, command)
    report = initialized(PredictiveBoundaryMonitor).evaluate(packet, command)
    for obligation in base.obligation_verdicts:
        if obligation != "geofence":
            assert report.obligation_verdicts[obligation] == base.obligation_verdicts[obligation]
    assert report.verdict is Verdict.VIOLATION  # Collision evidence remains a violation.


def test_prediction_does_not_turn_held_boundary_unknown_into_affirmative():
    monitor = initialized(PredictiveBoundaryMonitor)
    monitor.evaluate(make_packet(0, 1, (40.5, 0, -6)), make_command(0, 1, kind="hold"))
    report = monitor.evaluate(make_packet(1, 1.5, (40.5, 0, -6), has_state=False),
                              make_command(1, 1.5, kind="hold"))
    assert report.obligation_verdicts["geofence"] is Verdict.UNKNOWN
    assert not report.affirmative
    assert report.intervention == "return_to_launch"


def test_prediction_expands_with_measured_state_age():
    fresh = make_packet(0, 3, (37, 0, -6))
    stale = make_packet(0, 3, (37, 0, -6), state_age_s=2)
    command = make_command(0, 3, kind="hold")
    assert initialized(PredictiveBoundaryMonitor).evaluate(fresh, command).verdict is Verdict.PASS
    assert initialized(PredictiveBoundaryMonitor).evaluate(stale, command).verdict is Verdict.UNKNOWN


def test_immediate_abort_never_claims_safety_or_allows_inspection():
    monitor = initialized(ImmediateAbortMonitor)
    for k in range(3):
        report = monitor.evaluate(make_packet(k, k+.5, (0, 0, -6)),
                                  make_command(k, k+.5, kind="inspect_capture"))
        assert report.intervention == "abort"
        assert report.verdict is Verdict.UNKNOWN
        assert not report.affirmative
        assert set(report.obligation_verdicts.values()) == {Verdict.UNKNOWN}


def test_age_ablation_changes_only_authorization_when_record_is_old():
    token = granted_token(granted_at_s=1, validity_s=5)
    packet = make_packet(0, 4, (24.5, 0, -6), authorization=token)
    command = make_command(0, 4, kind="hold")
    original = initialized(AssumptionAwareMonitor).evaluate(packet, command)
    ablated = initialized(NoRecordAgeAuthorizationMonitor).evaluate(packet, command)
    assert original.obligation_verdicts["inspection_authorization"] is Verdict.UNKNOWN
    assert ablated.obligation_verdicts["inspection_authorization"] is Verdict.PASS
    for key in original.obligation_verdicts:
        if key != "inspection_authorization":
            assert ablated.obligation_verdicts[key] == original.obligation_verdicts[key]
    for key in original.assumption_verdicts:
        if key != "d_authorization_validity":
            assert ablated.assumption_verdicts[key] == original.assumption_verdicts[key]


@pytest.mark.parametrize("now, expected", [(1, "valid"), (5.75, "valid"),
                                           (5.8, "ambiguous"), (6, "invalid"), (7, "invalid")])
def test_age_ablation_retains_fixed_clock_margin_and_half_open_expiry(now, expected):
    packet = make_packet(0, now, (24.5, 0, -6),
                         authorization=granted_token(granted_at_s=1, validity_s=5))
    state, _ = initialized(NoRecordAgeAuthorizationMonitor)._classify_authorization(packet, now)
    assert state == expected


@pytest.mark.parametrize("status", [AuthorizationStatus.PENDING, AuthorizationStatus.ABSENT,
                                    AuthorizationStatus.DENIED])
def test_age_ablation_does_not_change_other_authorization_statuses(status):
    packet = make_packet(0, 2, (24.5, 0, -6), authorization=AuthorizationView(status=status))
    assert (initialized(NoRecordAgeAuthorizationMonitor)._classify_authorization(packet, 2)
            == initialized(AssumptionAwareMonitor)._classify_authorization(packet, 2))
