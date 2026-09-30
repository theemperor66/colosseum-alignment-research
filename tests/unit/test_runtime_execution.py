"""Runtime execution regressions: enacted interventions and one owner of simulated time.

Two findings from independent review are pinned here, against the fixture fake simulator over a real
socket:

* a ``hold`` intervention was logged but never enacted, so the guarded arm still flew the controller's
  command while the record claimed an intervention;
* the adapter's blocking ``hold``/``takeoff``/``land`` advanced the clock, and the runner then advanced
  it again, so one control step consumed two intervals and the first one held no truth samples.

Both are checked on the EXECUTED command and on measured simulator time, not on log labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from colosseum_assurance.config import AppConfig, PathsConfig
from colosseum_assurance.interfaces import MissionBrief
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.arms import Arm, mission_brief
from colosseum_assurance.runtime.episode import EpisodeRunner
from colosseum_assurance.scenario.manifest import build_manifest
from colosseum_assurance.schemas import ControlCommand, MonitorReport, ObservationPacket, Verdict
from colosseum_assurance.sim import build_adapter, fixture_fake_server

SHORT_HORIZON_S = 25.0  # long enough for the controller to leave the launch point


def test_perceived_target_alignment_uses_native_rotation_when_position_is_already_reached(monkeypatch):
    """Replay the nominal05 stall through controller, runner and adapter wire translation.

    A native short-path move can return without applying yaw. The test plant deliberately models
    that behavior instead of the older fixture's unconditional yaw update on every move command.
    """
    import math

    from tests.unit.test_controller import FakeVehicle, make_brief, make_packet, perceive

    from colosseum_assurance.config import EndpointConfig
    from colosseum_assurance.control.controller import InspectionController
    from colosseum_assurance.runtime.episode import _GuardState
    from colosseum_assurance.runtime.supervision import AuthorizationBroker
    from colosseum_assurance.schemas import AuthorizationView, Vec3
    from colosseum_assurance.sim.colosseum_adapter import ColosseumAdapter

    full = ProtocolConfig()
    vehicle = FakeVehicle(x=13.7536, y=-1.2151, z=-5.8402, yaw_rad=-.1696, landed=False)
    target = Vec3(x=27.1122, y=-1.1490, z=-6)
    initial_position = (vehicle.x, vehicle.y, vehicle.z)
    controller = InspectionController(full)
    controller.reset(make_brief())
    controller._set_phase("search_align", "replay_live_alignment_stall")
    adapter = ColosseumAdapter(EndpointConfig(), full)
    calls = []

    def send(key, method, *args):
        calls.append((method, args))
        if method == "moveToPosition":
            # Pinned moveOnPath can skip its loop when auto-lookahead already reaches the endpoint.
            assert math.dist(args[:3], initial_position) < .75
            return  # No yaw control is performed by this reached-position command.
        if method == "rotateToYaw":
            yaw_deg, timeout_s, margin_deg, _vehicle_name = args
            assert timeout_s <= 2 * full.mission.control_dt_s
            assert math.radians(margin_deg) < controller.params.align_tolerance_rad
            vehicle.yaw_rad = math.radians(yaw_deg)
            return
        raise AssertionError(method)

    monkeypatch.setattr(adapter, "_send", send)
    manifest = build_manifest(full, "fixture", "obs_nominal__sup_nominal", 0)
    runner = EpisodeRunner(adapter, full, AppConfig(), save_frames=False)
    broker = AuthorizationBroker(manifest.schedules, full.obligations)
    packet = make_packet(0, 20, vehicle.state(20), perceive(vehicle, target, 20), AuthorizationView())
    assert packet.depth.target_bearing_rad > controller.params.align_tolerance_rad
    command = controller.step(packet)
    runner._execute(command, _GuardState(), broker, manifest, 20, 0, [], {})
    next_packet = make_packet(1, 20.5, vehicle.state(20.5), perceive(vehicle, target, 20.5),
                              AuthorizationView())
    next_command = controller.step(next_packet)
    assert next_command.kind == "request_authorization"
    assert next_command.controller_phase == "request_authorization"
    assert [name for name, _args in calls] == ["rotateToYaw"]
    assert (vehicle.x, vehicle.y, vehicle.z) == initial_position


@pytest.fixture(scope="module")
def protocol() -> ProtocolConfig:
    base = ProtocolConfig()
    return base.model_copy(
        update={"mission": base.mission.model_copy(update={"episode_horizon_s": SHORT_HORIZON_S})},
        deep=True,
    )


class AlwaysHoldMonitor:
    """A guard that asks for a hold at every step. Nothing else about the arm changes."""

    monitor_id = "always_hold_stub"

    def reset(self, brief: MissionBrief) -> None:
        self.brief = brief

    def evaluate(self, observation: ObservationPacket, command: ControlCommand) -> MonitorReport:
        return MonitorReport(
            step_index=observation.step_index,
            sim_time_s=observation.receive_sim_time_s,
            monitor_id=self.monitor_id,
            verdict=Verdict.UNKNOWN,
            intervention="hold",
            rationale="stub guard: hold every step",
        )

    def describe(self) -> dict[str, Any]:
        return {"monitor_id": self.monitor_id, "purpose": "regression stub"}


def _run(protocol: ProtocolConfig, tmp_path, monitor=None):
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=None, save_frames=False)
            arm = None
            if monitor is not None:
                from colosseum_assurance.control.controller import InspectionController

                arm = Arm(
                    spec=protocol.arms.get("A1_policy_only"),
                    controller=InspectionController(protocol),
                    monitor=monitor,
                )
                # The brief is built inside run(); reset here only to satisfy a stub that needs it early.
                monitor.reset(mission_brief(protocol, manifest))
            return runner.run(manifest, "A1_policy_only", arm=arm), manifest
        finally:
            adapter.close()


def test_per_arm_scene_refusal_persists_without_inventing_a_flight_interval(
    protocol, tmp_path, monkeypatch,
):
    from colosseum_assurance.runtime.evidence import EvidenceWriter, load_attempted_runs
    from colosseum_assurance.sim.scene import SceneMismatch

    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol=protocol)
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=False)
            binding = runner.bind_scenario(manifest)

            def reject_scene(_manifest):
                raise SceneMismatch("synthetic actor appeared after matched binding")

            monkeypatch.setattr(adapter, "configure_scene", reject_scene)
            result = runner.run(manifest, "A0_unguarded", binding=binding)
        finally:
            adapter.close()
    assert result.attempt.status == "partial"
    assert result.attempt.error_type == "SceneMismatch"
    assert result.attempt.termination_reason == "setup_failed"
    assert result.record is not None and result.record.steps == []
    assert result.ledger is not None and result.ledger.samples == []
    assert result.ledger.truth_coverage_fraction == 0
    assert load_attempted_runs(writer.root/"attempted_runs.jsonl") == [result.attempt]
    assert result.episode_path and result.ledger_path


@pytest.mark.parametrize("failure_kind", ["AdapterError", "AdapterTimeout"])
@pytest.mark.parametrize("elapsed_s", [0.0, 0.01])
def test_immediate_adapter_failure_retains_attempt_and_diagnostic_truth_without_fake_coverage(
    protocol, tmp_path, monkeypatch, failure_kind, elapsed_s,
):
    from colosseum_assurance import interfaces
    from colosseum_assurance.runtime.evidence import EvidenceWriter, load_attempted_runs

    paths = PathsConfig(results_root=tmp_path)
    writer = EvidenceWriter(paths, "fixture", protocol.content_hash(), protocol=protocol)
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    with fixture_fake_server() as endpoint:
        config = AppConfig(endpoint=endpoint, paths=paths)
        adapter = build_adapter(config, protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=False)

            def fail_placement(*args):
                if elapsed_s:
                    adapter.sample_state()  # Establish this fixture scene's new clock origin first.
                    adapter.step(elapsed_s)
                raise getattr(interfaces, failure_kind)("synthetic immediate placement failure")

            monkeypatch.setattr(adapter, "set_start_pose", fail_placement)
            result = runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()
    assert result.attempt.error_type == failure_kind
    assert result.attempt.error_message == "synthetic immediate placement failure"
    assert result.attempt.status == ("timeout" if failure_kind == "AdapterTimeout" else "crashed")
    assert result.attempt.sim_duration_s == pytest.approx(elapsed_s)
    assert result.record.steps == []
    assert result.ledger.samples == []
    assert result.ledger.expected_sample_count == 0
    assert result.ledger.truth_coverage_fraction == 0
    end = result.ledger.events_of("episode_end")[0]
    diagnostic = end.payload["zero_interval_diagnostic_truth_samples"]
    assert len(diagnostic) == 1
    assert diagnostic[0]["sim_time_s"] == pytest.approx(elapsed_s)
    assert diagnostic[0]["source"] == "fixture_fake_ground_truth"
    assert load_attempted_runs(writer.root/"attempted_runs.jsonl") == [result.attempt]
    assert result.episode_path and result.ledger_path


def test_hold_intervention_is_enacted_not_only_logged(protocol: ProtocolConfig, tmp_path) -> None:
    result, _ = _run(protocol, tmp_path, monitor=AlwaysHoldMonitor())
    record = result.record
    assert record is not None and record.steps

    motion_kinds = {"move_to", "inspect_capture", "takeoff", "return_to_launch"}
    motion_steps = [s for s in record.steps if s.command.kind in motion_kinds]
    assert motion_steps, "the controller must have asked to move at least once in this episode"
    for step in motion_steps:
        assert step.executed_command is not None
        assert step.executed_command.kind == "hold", (
            f"step {step.step_index}: controller asked for {step.command.kind} and the guard asked for a "
            f"hold, but {step.executed_command.kind} was executed"
        )
        assert step.executed_command.issued_by == "guard"
    # A landing is deliberately not blocked by a hold: interrupting a descent is not the safer action.
    executed_kinds = {s.executed_command.kind for s in record.steps if s.executed_command}
    assert executed_kinds <= {"hold", "land"}
    assert all(step.monitor_report is not None for step in record.steps)

    ledger = result.ledger
    assert ledger is not None and ledger.samples
    # The vehicle must not travel while held. Small residual drift from the takeoff climb is allowed.
    start = ledger.samples[0].position
    travelled = max(sample.position.horizontal_distance_to(start) for sample in ledger.samples)
    assert travelled < 2.0, f"a held vehicle travelled {travelled:.2f} m horizontally"


def test_every_control_step_advances_exactly_one_dt(protocol: ProtocolConfig, tmp_path) -> None:
    result, _ = _run(protocol, tmp_path)
    record = result.record
    assert record is not None and record.steps
    dt = protocol.mission.control_dt_s
    advances = [step.provenance["sim_advance_s"] for step in record.steps]
    assert advances, "the runner must record how much simulated time each step consumed"
    for index, advance in enumerate(advances):
        assert advance == pytest.approx(dt, abs=1e-6), (
            f"step {index} advanced {advance} s instead of {dt} s: time is advanced twice per step"
        )


def test_truth_sampling_has_no_gap_larger_than_the_protocol_allows(
    protocol: ProtocolConfig, tmp_path
) -> None:
    result, _ = _run(protocol, tmp_path)
    ledger = result.ledger
    assert ledger is not None and len(ledger.samples) > 10
    limit = protocol.simulation.max_permitted_truth_gap_s
    assert ledger.max_sample_gap_s is not None
    assert ledger.max_sample_gap_s <= limit, (
        f"largest truth gap {ledger.max_sample_gap_s} s exceeds the protocol limit {limit} s, so the "
        "evaluator would have to return unknown for the blind interval"
    )
    assert ledger.truth_coverage_fraction >= 0.95
    per_step = [step.provenance["truth_samples_this_step"] for step in record_steps(result)]
    expected = round(protocol.mission.control_dt_s / protocol.simulation.truth_sample_interval_s)
    assert set(per_step) == {expected}


def record_steps(result):
    assert result.record is not None
    return result.record.steps


# --------------------------------------------------------------------------------------
# Continuation-review regressions: setup failure, evidence retention, binding order
# --------------------------------------------------------------------------------------
class UnreachableAltitudeAdapter:
    """Wraps a real adapter and drops every climb command, so the vehicle never leaves the ground.

    This is the software counterexample from the review: one bounded climb command that expires, or a
    vehicle that simply does not climb, must NOT produce an ordinary research episode.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._inner, name)

    def move_to(self, *args, **kwargs) -> None:
        return None

    def issue_takeoff(self, *args, **kwargs) -> None:
        return None


def test_a_failed_takeoff_ends_the_episode_incomplete(protocol: ProtocolConfig, tmp_path) -> None:
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(
                UnreachableAltitudeAdapter(adapter), protocol, config, writer=None, save_frames=False
            )
            result = runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()

    assert result.record is not None
    termination = result.record.termination
    assert termination.reason == "setup_failed"
    assert termination.reached_terminal_state is False, "a failed setup is incomplete evidence"
    assert termination.completed_mission is False
    assert result.attempt.status != "completed"
    assert "cruise altitude" in termination.detail
    assert result.ledger is not None
    kinds = [event.kind for event in result.ledger.events]
    assert "simulator_error" in kinds, "the failed setup must be visible in the privileged ledger"


def test_recorded_evidence_is_never_overwritten_by_a_repeat(protocol: ProtocolConfig, tmp_path) -> None:
    """A retry used to overwrite the first episode while appending a second attempt row."""
    from colosseum_assurance.runtime.evidence import EvidenceExists, EvidenceWriter

    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    writer = EvidenceWriter(
        paths=config.paths, run_class="fixture", protocol_hash=protocol.content_hash(),
        protocol=protocol,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=writer, save_frames=False)
            first = runner.run(manifest, "A0_unguarded")
            assert first.episode_path is not None
            before = Path(first.episode_path).read_text(encoding="utf-8")
            with pytest.raises(EvidenceExists):
                runner.run(manifest, "A0_unguarded")
            after = Path(first.episode_path).read_text(encoding="utf-8")
        finally:
            adapter.close()

    assert before == after, "the first episode's evidence must survive a refused repeat"
    attempts = (writer.root / "attempted_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(attempts) == 1, "a refused repeat must not append a second attempt row"


def test_the_run_tree_persists_the_protocol_it_was_produced_under(
    protocol: ProtocolConfig, tmp_path
) -> None:
    """Post-processing must score a run under its own protocol, not under current defaults."""
    from colosseum_assurance.runtime.evidence import (
        EvidenceWriter,
        ProtocolMismatch,
        load_run_protocol,
    )

    writer = EvidenceWriter(
        paths=PathsConfig(results_root=tmp_path), run_class="fixture",
        protocol_hash=protocol.content_hash(), protocol=protocol,
    )
    loaded, provenance = load_run_protocol(writer.root)
    assert provenance["source"] == "run_tree"
    assert loaded.content_hash() == protocol.content_hash()
    assert loaded.mission.episode_horizon_s == protocol.mission.episode_horizon_s

    other = ProtocolConfig()
    assert other.content_hash() != protocol.content_hash()
    with pytest.raises(ProtocolMismatch):
        load_run_protocol(writer.root, other)


def test_binding_is_established_once_and_every_arm_flies_it(protocol: ProtocolConfig, tmp_path) -> None:
    """The matched arm set must share one verified world, briefed from the geometry actually flown."""
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=None, save_frames=False)
            binding = runner.bind_scenario(manifest)
            flown = [
                runner.run(manifest, arm_id, binding=binding).record
                for arm_id in ("A0_unguarded", "A1_policy_only")
            ]
        finally:
            adapter.close()

    assert binding.effective.content_hash() == binding.requested.content_hash() or binding.changed
    hashes = {record.steps[0].provenance["flown_manifest_hash"] for record in flown if record}
    assert len(hashes) == 1, f"arms flew different geometry: {hashes}"
    assert hashes == {binding.effective.content_hash()}


def test_the_climb_command_is_refreshed_so_it_cannot_expire(protocol: ProtocolConfig, tmp_path) -> None:
    """Upstream `moveToPosition` is bounded by the timeout it is given, so one 0.5 s command expires.

    The fixture fake now expires commands on its own clock, so a runner that issues a single bounded
    climb and then waits would stop steering after half a second. The regression asserts that the
    runner refreshed the climb and that the vehicle actually reached cruise altitude.
    """
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=None, save_frames=False)
            result = runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()

    assert result.ledger is not None
    takeoff = [e for e in result.ledger.events if e.kind == "takeoff_complete"]
    assert takeoff, "a successful climb must record takeoff_complete"
    payload = takeoff[0].payload
    assert payload["reached"] is True
    assert payload["climb_command_refreshes"] >= 1, (
        "the climb must be re-issued at least once; a single bounded command expires"
    )
    assert payload["final_height_m"] >= protocol.mission.cruise_altitude_m - payload["tolerance_m"]
    assert result.record is not None
    assert result.record.termination.reason != "setup_failed"


def test_measured_clock_facts_are_recorded_with_every_episode(
    protocol: ProtocolConfig, tmp_path
) -> None:
    """A degraded or fallback stepping mode must be visible in the evidence, not inferred."""
    manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(adapter, protocol, config, writer=None, save_frames=False)
            result = runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()

    assert result.record is not None and result.ledger is not None
    report = result.record.timing_report
    assert report, "the episode record must carry the adapter's measured clock facts"
    assert report["mode_used"] in {"paused_continue_for_time", "wall_clock"}
    assert report["steps_taken"] > 0
    assert "clock_type" in report and "last_measured_advance_s" in report
    assert result.ledger.timing_report == report


# --------------------------------------------------------------------------------------
# Image freshness: acquisition time, never processing time
# --------------------------------------------------------------------------------------
class StaleCameraAdapter:
    """Delegates to a real adapter but hands back frames acquired earlier than now.

    ``offset_s`` backdates the acquisition timestamp; ``drop_timestamp`` removes it entirely. Both keep
    the pixels, so the frame stays nonempty: the question is whether the client treats old or
    unknown-age pixels as fresh evidence.
    """

    def __init__(self, inner, offset_s: float = 0.0, drop_timestamp: bool = False) -> None:
        self._inner = inner
        self._offset_s = offset_s
        self._drop_timestamp = drop_timestamp

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._inner, name)

    def capture(self, kinds=("rgb", "depth"), save_prefix=None):
        frames = self._inner.capture(kinds=kinds, save_prefix=save_prefix)
        out = {}
        for name, frame in frames.items():
            ref = frame.ref
            if self._drop_timestamp:
                ref = ref.model_copy(update={"sim_time_s": None, "acquisition_time_known": False})
            elif self._offset_s:
                aged = max(0.0, (ref.sim_time_s or 0.0) - self._offset_s)
                ref = ref.model_copy(update={"sim_time_s": aged})
            frame.ref = ref
            out[name] = frame
        return out


def _run_with_camera(protocol, tmp_path, **camera_kwargs):
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(
                StaleCameraAdapter(adapter, **camera_kwargs), protocol, config,
                writer=None, save_frames=False,
            )
            manifest = build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0)
            return runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()


def test_a_stale_frame_keeps_its_native_age_in_the_observation(protocol: ProtocolConfig, tmp_path):
    """A frame acquired 8 s ago must not look zero seconds old because it was processed now."""
    offset = 8.0
    result = _run_with_camera(protocol, tmp_path, offset_s=offset)
    record = result.record
    assert record is not None

    aged = [
        step for step in record.steps
        if step.observation.depth is not None and step.sim_time_s >= offset + 1.0
    ]
    assert aged, "no step carried a depth summary late enough to measure native staleness"
    for step in aged[:5]:
        depth = step.observation.depth
        assert depth is not None
        measured_age = step.observation.receive_sim_time_s - depth.sim_time_s
        assert measured_age >= offset - 0.75, (
            f"step {step.step_index}: depth age {measured_age:.2f} s hides the camera's "
            f"{offset:.1f} s native staleness"
        )
        # The summary must carry the acquisition time, not the processing time.
        assert depth.sim_time_s < step.observation.receive_sim_time_s


def test_a_frame_without_an_acquisition_time_cannot_enter_perception(
    protocol: ProtocolConfig, tmp_path
):
    """Nonempty pixels of unknown age are not evidence, and must never be dated to now."""
    result = _run_with_camera(protocol, tmp_path, drop_timestamp=True)
    record = result.record
    assert record is not None and record.steps

    for step in record.steps:
        observation = step.observation
        assert observation.depth is None, (
            f"step {step.step_index} accepted a frame of unknown age as perception evidence"
        )
        assert observation.rgb is None
        assert observation.sensor_health.depth_available is False
        assert observation.sensor_health.depth_age_s is None


def test_an_unstamped_frame_is_marked_unusable_by_the_adapter():
    """The schema itself refuses to call an untimed frame usable evidence."""
    from colosseum_assurance.schemas import FrameRef

    untimed = FrameRef(
        kind="depth", camera_name="front_center", sim_time_s=None, acquisition_time_known=False,
        width=8, height=8, nonzero_fraction=0.9, max_value=12.0,
    )
    assert untimed.is_nonempty, "the pixels are real; only the age is unknown"
    assert not untimed.is_usable_evidence
    with pytest.raises(ValidationError):
        FrameRef(
            kind="depth", camera_name="front_center", sim_time_s=None, acquisition_time_known=True,
            width=8, height=8,
        )


def test_reading_an_older_frame_does_not_rewind_the_world_clock(protocol: ProtocolConfig, tmp_path):
    """Capture-time conversion must not touch the authoritative clock or count a false regression."""
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            adapter.reset()
            adapter.wait_until_ready()
            adapter.acquire_control()
            adapter.arm()
            for _ in range(20):
                adapter.step(0.5)
            before_time = adapter.sim_time_s()
            before_regressions = adapter.clock_regressions
            stamp_ns = int((before_time - 5.0) * 1_000_000_000) + (adapter._time_origin_ns or 0)
            converted = adapter._to_episode_time(stamp_ns)
            after_time = adapter.sim_time_s()
        finally:
            adapter.close()

    assert converted == pytest.approx(before_time - 5.0, abs=1e-3)
    assert after_time == pytest.approx(before_time), "converting a capture time moved the world clock"
    assert adapter.clock_regressions == before_regressions, "a false clock regression was counted"


# --------------------------------------------------------------------------------------
# Hold with a heading is a rotate-in-place, not a hover
# --------------------------------------------------------------------------------------
def test_a_full_mission_completes_through_the_production_runner(protocol: ProtocolConfig, tmp_path):
    """The whole mission must finish: inspect, turn, return, land.

    A hover does not rotate. When the runner dropped the heading from a controller hold, the vehicle
    sat in `returning_home:turning_to_face_goal` for 209 consecutive steps with its yaw unchanged and
    the episode ended at the horizon. This drives the production runner over the fixture RPC wire and
    asserts the mission actually completed, not merely that the horizon was reached.
    """
    full = ProtocolConfig()  # the full horizon, not the shortened one used elsewhere in this file
    manifest = build_manifest(full, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), full)
        try:
            runner = EpisodeRunner(adapter, full, config, writer=None, save_frames=False)
            result = runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()

    record, ledger = result.record, result.ledger
    assert record is not None and ledger is not None
    assert record.termination.reason == "mission_complete", (
        f"the mission did not finish: {record.termination.reason} after {len(record.steps)} steps, "
        f"phases={sorted({s.controller_state.get('phase') for s in record.steps})}"
    )
    assert record.termination.completed_mission is True

    # Task IDs can repeat across processes; the privileged snapshot is linked to this exact episode.
    landing = ledger.events_of("episode_end")[0].payload["native_landing_handshake"]
    assert landing["episode_id"] == record.episode_id
    assert landing["report"]["task_id"] >= 1
    assert any(row["kind"] == "native_land_issued" for row in landing["report"]["events"])

    # The heading actually changed, and the vehicle came home.
    yaws = [s.observation.state.yaw_rad for s in record.steps if s.observation.state is not None]
    assert max(abs(y - yaws[0]) for y in yaws) > 1.0, "the vehicle never turned"
    home = full.mission.home
    assert ledger.samples[-1].position.horizontal_distance_to(home) <= full.mission.return_tolerance_m

    # The inspection really happened before the return.
    captures = [e for e in ledger.events if e.kind == "inspection_capture_performed"]
    assert len(captures) >= full.mission.required_inspection_captures

    # A turning hold must not sit unchanged for hundreds of steps.
    turning = [
        s for s in record.steps
        if s.command.kind == "hold" and "turning_to_face_goal" in (s.command.reason or "")
    ]
    assert len(turning) < 20, f"{len(turning)} turning holds: the heading command is not being enacted"


def test_a_controller_hold_with_a_heading_commands_a_rotation(protocol: ProtocolConfig, tmp_path):
    """The rotation reaches the simulator, and a guard hold stays a pure hover."""
    calls: list[tuple[str, float | None]] = []

    class RecordingAdapter:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):  # pragma: no cover - thin delegation
            return getattr(self._inner, name)

        def issue_hold(self):
            calls.append(("hold", None))
            return self._inner.issue_hold()

        def issue_rotate_to_yaw(self, yaw_rad, timeout_s=None, margin_deg=5.0):
            calls.append(("rotate", float(yaw_rad)))
            return self._inner.issue_rotate_to_yaw(yaw_rad, timeout_s=timeout_s, margin_deg=margin_deg)

    full = ProtocolConfig()
    manifest = build_manifest(full, "fixture", "obs_nominal__sup_nominal", 0)
    config = AppConfig(
        run_class="fixture", paths=PathsConfig(results_root=tmp_path),
        allow_fixture_fake=True, require_live_simulator=False,
    )
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), full)
        try:
            runner = EpisodeRunner(RecordingAdapter(adapter), full, config, writer=None,
                                   save_frames=False)
            runner.run(manifest, "A0_unguarded")
        finally:
            adapter.close()

    rotations = [yaw for kind, yaw in calls if kind == "rotate"]
    assert rotations, "no rotation command reached the simulator"

    # A guard hold must not rotate: the guard suppresses movement, it does not re-aim the vehicle.
    calls.clear()
    with fixture_fake_server() as endpoint:
        adapter = build_adapter(config.model_copy(update={"endpoint": endpoint}), protocol)
        try:
            runner = EpisodeRunner(RecordingAdapter(adapter), protocol, config, writer=None,
                                   save_frames=False)
            arm = Arm(
                spec=protocol.arms.get("A1_policy_only"),
                controller=__import__(
                    "colosseum_assurance.control.controller", fromlist=["InspectionController"]
                ).InspectionController(protocol),
                monitor=AlwaysHoldMonitor(),
            )
            arm.monitor.reset(mission_brief(protocol, build_manifest(
                protocol, "fixture", "obs_nominal__sup_nominal", 0)))
            runner.run(build_manifest(protocol, "fixture", "obs_nominal__sup_nominal", 0),
                       "A1_policy_only", arm=arm)
        finally:
            adapter.close()
    assert all(kind == "hold" for kind, _ in calls), (
        f"a guard hold issued a rotation: {[k for k, _ in calls if k != 'hold']}"
    )
