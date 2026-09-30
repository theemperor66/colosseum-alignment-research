"""The real :class:`ColosseumAdapter` driven against the fixture fake over a real TCP socket.

Nothing is mocked here. The adapter under test is exactly the adapter that would talk to a live
simulator; only the server on the other end is a test double, and it says so. That is why these tests
can check wire encoding, timeout conversion and provenance handling at once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from colosseum_assurance.config import AppConfig, EndpointConfig
from colosseum_assurance.interfaces import AdapterError, AdapterTimeout, SimAdapter
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ObstacleSpec, build_manifest
from colosseum_assurance.schemas import Vec3
from colosseum_assurance.sim.colosseum_adapter import ColosseumAdapter, build_adapter
from colosseum_assurance.sim.diagnostics import (
    FIXTURE_GATE_REASON,
    run_diagnostics,
    run_live_readiness_gate,
)
from colosseum_assurance.sim.fixture_fake import (
    FixtureFakeServer,
    FixtureFakeSimulator,
    fixture_fake_server,
)
from colosseum_assurance.sim.identity import FixtureProbeResult, decide_provenance
from colosseum_assurance.sim.scene import verify_scene

PROTOCOL = ProtocolConfig()


@pytest.fixture()
def fake() -> Iterator[tuple[EndpointConfig, FixtureFakeSimulator]]:
    simulator = FixtureFakeSimulator(image_width=64, image_height=36)
    server = FixtureFakeServer(simulator).start()
    try:
        yield server.endpoint, simulator
    finally:
        server.stop()


@pytest.fixture()
def adapter(fake: tuple[EndpointConfig, FixtureFakeSimulator]) -> Iterator[ColosseumAdapter]:
    endpoint, _ = fake
    client = ColosseumAdapter(endpoint, PROTOCOL)
    client.connect()
    try:
        yield client
    finally:
        client.close()


def _flying(client: ColosseumAdapter, altitude_m: float = 4.0) -> None:
    client.reset()
    client.wait_until_ready(timeout_s=10.0)
    client.acquire_control()
    client.takeoff(altitude_m, timeout_s=30.0)


# ------------------------------------------------------------------ identity and provenance
def test_adapter_satisfies_the_simadapter_protocol(adapter: ColosseumAdapter) -> None:
    assert isinstance(adapter, SimAdapter)


def test_handshake_downgrades_provenance_to_fixture_fake(adapter: ColosseumAdapter) -> None:
    identity = adapter.identity()
    assert identity.provenance == "fixture_fake"
    assert identity.is_live is False
    assert identity.server_version == 1
    assert identity.min_required_client_version == 1
    assert identity.scene_object_count and identity.scene_object_count > 0
    assert identity.scene_signature and identity.scene_signature.startswith("sha256:")
    assert identity.settings_digest and identity.settings_digest.startswith("sha256:")
    assert identity.engine_version is None, "an engine version must never be invented"
    assert "fixture fake" in identity.notes


def test_provenance_rule_requires_the_probe_to_be_rejected() -> None:
    rejected = FixtureProbeResult(rejected_with_rpc_error=True, detail="unknown method")
    assert (
        decide_provenance(ping_ok=True, probe=rejected, server_version=1)[0]
        == "airsim_compatible_unverified"
    ), "a rejected probe proves an RPC surface, not a Colosseum build"
    affirmative = FixtureProbeResult(answered_affirmative=True, detail="fixture")
    assert decide_provenance(ping_ok=True, probe=affirmative, server_version=1)[0] == "fixture_fake"
    # A server that answers the private probe with anything else is not proven live.
    other = FixtureProbeResult(answered_other=True, detail="answered 0")
    assert decide_provenance(ping_ok=True, probe=other, server_version=1)[0] == "unverified"
    assert decide_provenance(ping_ok=False, probe=rejected, server_version=1)[0] == "unverified"
    assert decide_provenance(ping_ok=True, probe=rejected, server_version=None)[0] == "unverified"


# ------------------------------------------------------------------ session
def test_reset_drops_api_control_and_a_later_command_is_refused(adapter: ColosseumAdapter) -> None:
    adapter.reset()
    adapter.wait_until_ready(timeout_s=10.0)
    adapter.acquire_control()
    adapter.reset()
    assert adapter.sample_state().api_control_enabled is False
    adapter.move_to(Vec3(x=3.0, y=0.0, z=-3.0), 2.0, 1.0)
    with pytest.raises(AdapterError, match="API control"):
        adapter.step(PROTOCOL.mission.control_dt_s)


def test_control_is_acquired_and_released_with_verification(
    adapter: ColosseumAdapter, fake: tuple[EndpointConfig, FixtureFakeSimulator]
) -> None:
    _, simulator = fake
    adapter.reset()
    adapter.wait_until_ready(timeout_s=10.0)
    adapter.acquire_control()
    assert simulator.api_control is True
    assert simulator.armed is True
    adapter.release_control()
    assert simulator.api_control is False
    assert simulator.armed is False


def test_disarm_acknowledgement_expires_on_arming_reset_and_release(adapter: ColosseumAdapter):
    adapter.acquire_control(arm=False)
    assert adapter._disarm_acknowledgement["result"] is True
    adapter.arm()
    assert adapter._disarm_acknowledgement is None
    adapter.disarm()
    assert adapter._disarm_acknowledgement["result"] is True
    adapter.reset()
    assert adapter._disarm_acknowledgement is None
    adapter.acquire_control(arm=False)
    adapter.release_control()
    assert adapter._disarm_acknowledgement is None


def test_wait_until_ready_reports_a_remedy_when_the_vehicle_never_settles() -> None:
    with fixture_fake_server(image_size=(32, 18), fail_methods={"isApiControlEnabled": "denied"}
                             ) as endpoint:
        client = ColosseumAdapter(endpoint.model_copy(update={"reset_timeout_s": 1.0}), PROTOCOL)
        client.connect()
        try:
            with pytest.raises(AdapterError, match="not ready"):
                client.wait_until_ready(timeout_s=1.0)
        finally:
            client.close()


# ------------------------------------------------------------------ motion and stepping
def test_bounded_move_changes_position(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 4.0)
    start = adapter.sample_state().position
    target = Vec3(x=start.x + 8.0, y=start.y, z=start.z)
    positions = [start]
    for index in range(10):
        if index % 2 == 0:
            adapter.move_to(target, PROTOCOL.mission.cruise_speed_mps, 2.0)
        adapter.step(PROTOCOL.mission.control_dt_s)
        positions.append(adapter.sample_state().position)
    travelled = max(start.distance_to(p) for p in positions)
    assert travelled > 3.0, f"the commanded move only produced {travelled:.2f} m of motion"
    assert positions[-1].x > start.x + 3.0


def test_move_speed_is_clamped(adapter: ColosseumAdapter,
                               fake: tuple[EndpointConfig, FixtureFakeSimulator]) -> None:
    _, simulator = fake
    _flying(adapter, 4.0)
    adapter.move_to(Vec3(x=20.0, y=0.0, z=-4.0), 500.0, 2.0)
    adapter.step(PROTOCOL.mission.control_dt_s)
    assert simulator.command_speed == pytest.approx(adapter.max_speed_mps)
    assert adapter.max_speed_mps < 500.0


def test_move_duration_is_bounded_by_the_protocol(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 4.0)
    adapter.move_to(Vec3(x=20.0, y=0.0, z=-4.0), 2.0, 10_000.0)
    adapter.step(PROTOCOL.mission.control_dt_s)
    # The duration reaches the server as moveToPosition's timeout_sec, clamped to the protocol bound.
    assert PROTOCOL.simulation.max_command_duration_s < 10_000.0


def test_step_uses_paused_continue_for_time_and_advances_the_clock(adapter: ColosseumAdapter) -> None:
    adapter.reset()
    adapter.wait_until_ready(timeout_s=10.0)
    before = adapter.sim_time_s()
    for _ in range(4):
        adapter.step(0.25)
    assert adapter.sim_time_s() - before == pytest.approx(1.0, abs=1e-6)
    report = adapter.stepping_report()
    assert report["mode_used"] == "paused_continue_for_time"
    assert report["fallback_reason"] is None


def test_hold_advances_time_without_moving(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 4.0)
    before = adapter.sample_state()
    adapter.hold(1.5)
    after = adapter.sample_state()
    assert after.sim_time_s - before.sim_time_s == pytest.approx(1.5, abs=0.05)
    assert before.position.distance_to(after.position) < 0.6


def test_land_returns_the_vehicle_to_the_ground(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 3.0)
    adapter.land(timeout_s=30.0)
    assert adapter.sample_state().landed is True


# ------------------------------------------------------------------ observation
def test_state_sample_carries_no_privileged_collision_data(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 4.0)
    state = adapter.sample_state()
    fields = set(type(state).model_fields)
    assert not {field for field in fields if "collision" in field}
    truth = adapter.sample_truth()
    assert truth.source == "fixture_fake_ground_truth"
    assert hasattr(truth, "collision_active")


def test_truth_sample_reports_a_collision_the_state_sample_cannot_see(
    adapter: ColosseumAdapter,
) -> None:
    adapter.reset()
    adapter.wait_until_ready(timeout_s=10.0)
    adapter.acquire_control()
    # Start beyond the occluding wall so the only body on the path is the tower itself.
    adapter.set_start_pose(Vec3(x=15.0, y=0.0, z=-7.0), 0.0)
    assert adapter.sample_truth().collision_active is False
    target = Vec3(x=20.0, y=0.0, z=-7.0)   # the centre of the fixture inspection tower
    for _ in range(24):
        adapter.move_to(target, 4.0, 2.0)
        adapter.step(0.5)
    truth = adapter.sample_truth()
    assert truth.collision_active is True, "flying into the tower produced no privileged collision"
    assert truth.collision_count >= 1
    assert truth.collision_object == "fixture_inspection_tower"
    assert truth.min_obstacle_clearance_m is not None
    assert truth.min_obstacle_clearance_m < 0.5


def test_capture_returns_typed_arrays_for_rgb_and_depth(adapter: ColosseumAdapter) -> None:
    _flying(adapter, 4.0)
    frames = adapter.capture(("rgb", "depth"))
    assert set(frames) == {"rgb", "depth"}

    depth = frames["depth"]
    assert depth.array is not None
    assert depth.array.dtype == np.float32
    assert depth.array.shape == (36, 64)
    assert depth.ref.kind == "depth"
    assert depth.ref.is_nonempty
    assert depth.ref.min_value is not None and depth.ref.min_value > 0.0
    assert depth.ref.max_value is not None and depth.ref.max_value <= 100.0

    rgb = frames["rgb"]
    assert rgb.array is not None
    assert rgb.array.dtype == np.uint8
    assert rgb.array.shape == (36, 64, 3)
    assert rgb.ref.kind == "rgb"
    assert rgb.ref.is_nonempty
    assert rgb.ref.nonzero_fraction is not None and rgb.ref.nonzero_fraction > 0.0


def test_capture_can_save_frames_to_disk(adapter: ColosseumAdapter, tmp_path: Path) -> None:
    adapter.frames_dir = tmp_path
    _flying(adapter, 4.0)
    frames = adapter.capture(("rgb", "depth"), save_prefix="ep0001_step0007")
    rgb_path = Path(frames["rgb"].ref.path or "")
    depth_path = Path(frames["depth"].ref.path or "")
    assert rgb_path.is_file() and rgb_path.suffix == ".png"
    assert depth_path.is_file() and depth_path.suffix == ".npy"
    assert frames["rgb"].ref.pixels_as == "png"
    assert frames["depth"].ref.pixels_as == "npy"
    restored = np.load(depth_path)
    assert restored.shape == (36, 64)
    assert np.allclose(restored, frames["depth"].array)


def test_capture_rejects_unknown_kinds(adapter: ColosseumAdapter) -> None:
    with pytest.raises(ValueError, match="unsupported capture kinds"):
        adapter.capture(("thermal",))


# ------------------------------------------------------------------ wire decoding
def test_structures_encoded_as_maps_are_also_decoded() -> None:
    """Upstream sends positional arrays; a map-sending variant must not silently corrupt data."""
    simulator = FixtureFakeSimulator(image_width=32, image_height=18, encoding="map")
    server = FixtureFakeServer(simulator).start()
    try:
        client = ColosseumAdapter(server.endpoint, PROTOCOL)
        client.connect()
        try:
            client.reset()
            client.wait_until_ready(timeout_s=10.0)
            state = client.sample_state()
            assert state.position.as_tuple() == (0.0, 0.0, 0.0)
            frames = client.capture(("depth",))
            assert frames["depth"].array is not None
            assert frames["depth"].array.shape == (18, 32)
        finally:
            client.close()
    finally:
        server.stop()


def test_a_wrong_structure_length_is_a_clear_adapter_error() -> None:
    with pytest.raises(AdapterError, match="expected 6 positional fields"):
        ColosseumAdapter._struct([1.0, 2.0], (
            "position", "orientation", "linear_velocity", "angular_velocity",
            "linear_acceleration", "angular_acceleration"), "KinematicsState")


def test_rpc_timeout_becomes_adapter_timeout() -> None:
    with fixture_fake_server(image_size=(32, 18), hang_methods={"getMultirotorState"}) as endpoint:
        client = ColosseumAdapter(endpoint.model_copy(update={"rpc_timeout_s": 0.3}), PROTOCOL)
        client.connect()
        try:
            with pytest.raises(AdapterTimeout, match="getMultirotorState did not answer"):
                client.sample_state()
            # The connection survives an abandoned call: other RPCs still work.
            assert client.ping() is True
        finally:
            client.close()


def test_connection_failure_explains_the_remedy() -> None:
    endpoint = EndpointConfig(host="127.0.0.1", port=1, connect_timeout_s=1.0)
    with pytest.raises(AdapterError) as excinfo:
        ColosseumAdapter(endpoint, PROTOCOL).connect()
    assert "ssh -N -L" in excinfo.value.remedy


def test_an_outside_reset_is_detected_as_a_clock_regression(
    adapter: ColosseumAdapter, fake: tuple[EndpointConfig, FixtureFakeSimulator]
) -> None:
    """Someone else resetting the simulator mid-episode must not pass unnoticed."""
    from colosseum_assurance.rpc.msgpack_rpc import MsgpackRpcClient

    endpoint, _ = fake
    _flying(adapter, 4.0)
    for _ in range(4):
        adapter.step(0.5)
    assert adapter.sample_state().sim_time_s > 1.0
    assert adapter.clock_regressions == 0

    with MsgpackRpcClient(endpoint.host, endpoint.port, connect_timeout_s=3.0,
                          call_timeout_s=5.0) as intruder:
        intruder.call("reset")          # a second client resets the world behind our back
    adapter.sample_state()
    assert adapter.clock_regressions == 1
    assert adapter.stepping_report()["clock_regressions"] == 1


# ------------------------------------------------------------------ scene
def test_configure_scene_loads_the_manifest_into_the_fixture(adapter: ColosseumAdapter) -> None:
    manifest = build_manifest(PROTOCOL, "fixture", PROTOCOL.cells()[0]["cell_id"], 0)
    result = adapter.configure_scene(manifest)
    assert result["mode"] == "fixture_loaded"
    assert result["obstacles"] == len(manifest.obstacles)
    assert result["provenance"] == "fixture_fake"
    listed = adapter.list_scene_objects(".*")
    for obstacle in manifest.obstacles:
        assert obstacle.name in listed


def test_verify_scene_finds_expected_actors_and_reports_missing_ones(
    adapter: ColosseumAdapter,
) -> None:
    manifest = build_manifest(PROTOCOL, "fixture", PROTOCOL.cells()[0]["cell_id"], 0)
    adapter.configure_scene(manifest)
    report = verify_scene(adapter, manifest)
    assert report.ok is True
    assert report.missing_actors == []
    assert report.scene_object_count > 0
    assert set(report.matched_actors) == {
        obstacle.unreal_actor_tag or obstacle.name for obstacle in manifest.obstacles
    }

    ghost = ObstacleSpec(
        name="never_built_hangar", kind="building", center=Vec3(x=5.0, y=5.0, z=-3.0),
        extent=Vec3(x=2.0, y=2.0, z=3.0), unreal_actor_tag="NeverBuiltHangar",
    )
    wrong = manifest.model_copy(update={"obstacles": [*manifest.obstacles, ghost]})
    broken = verify_scene(adapter, wrong)
    assert broken.ok is False
    assert broken.missing_actors == ["NeverBuiltHangar"]
    assert "not the scene the manifest describes" in broken.detail


def test_scene_name_is_refused_because_no_rpc_returns_it(adapter: ColosseumAdapter) -> None:
    with pytest.raises(NotImplementedError, match="no verified RPC returns the current level name"):
        adapter.scene_name()
    assert adapter.identity().scene_name is None


def test_set_visibility_uses_the_verified_weather_api(
    adapter: ColosseumAdapter, fake: tuple[EndpointConfig, FixtureFakeSimulator]
) -> None:
    _, simulator = fake
    result = adapter.set_visibility("reduced")
    assert result == {"visibility": "reduced", "weather_parameter": "Fog", "value": 0.5}
    assert simulator.weather_enabled is True
    assert simulator.weather[7] == pytest.approx(0.5)
    adapter.set_visibility("clear")
    assert simulator.weather[7] == pytest.approx(0.0)
    with pytest.raises(NotImplementedError, match="no verified simulator mapping"):
        adapter.set_visibility("sandstorm")


def test_verify_scene_checks_the_tag_route_as_well(adapter: ColosseumAdapter) -> None:
    manifest = build_manifest(PROTOCOL, "fixture", PROTOCOL.cells()[0]["cell_id"], 0)
    adapter.configure_scene(manifest)
    report = verify_scene(adapter, manifest)
    assert report.tag_route_available is True
    assert report.tag_route_missing == []
    assert report.route_disagreement == []
    assert report.tag_route_actors == sorted(
        obstacle.unreal_actor_tag for obstacle in manifest.obstacles if obstacle.unreal_actor_tag
    )


# ------------------------------------------------------------------ diagnostics and the live gate
def test_diagnostics_pass_against_the_fixture_but_name_it(fake: tuple[EndpointConfig,
                                                                     FixtureFakeSimulator]) -> None:
    endpoint, _ = fake
    report = run_diagnostics(AppConfig(endpoint=endpoint, run_class="fixture"), PROTOCOL)
    assert report.ok, report.render_text()
    assert report.provenance == "fixture_fake"
    ids = [check.id for check in report.checks]
    assert ids == [
        "configuration", "tcp_reachable", "rpc_ping", "version_handshake", "fixture_fake_probe",
        "reset", "api_control", "state_sample", "rgb_capture", "depth_capture", "scene_objects",
        "stepping_mode",
    ]
    probe = next(check for check in report.checks if check.id == "fixture_fake_probe")
    assert "never experimental evidence" in probe.detail


def test_diagnostics_explain_a_closed_port() -> None:
    config = AppConfig(endpoint=EndpointConfig(host="127.0.0.1", port=1, connect_timeout_s=1.0))
    report = run_diagnostics(config, PROTOCOL)
    assert not report.ok
    failure = report.first_failure()
    assert failure is not None and failure.id == "tcp_reachable"
    assert "refused" in failure.detail
    assert "ssh -N -L" in failure.remedy
    skipped = [check.id for check in report.checks if check.status == "skip"]
    assert "rpc_ping" in skipped and "depth_capture" in skipped


def test_diagnostics_distinguish_a_dead_tunnel_from_a_closed_port() -> None:
    """An SSH forward with no simulator behind it accepts TCP and then drops the stream.

    That is a completely different failure from "connection refused" and needs a different fix, so the
    doctor must not collapse the two into one message.
    """
    import socketserver
    import threading

    class _DeadTunnelHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.close()

    class _DeadTunnel(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _DeadTunnel(("127.0.0.1", 0), _DeadTunnelHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        endpoint = EndpointConfig(host="127.0.0.1", port=int(server.server_address[1]),
                                  connect_timeout_s=2.0, rpc_timeout_s=2.0)
        report = run_diagnostics(AppConfig(endpoint=endpoint), PROTOCOL)
        tcp_check = next(check for check in report.checks if check.id == "tcp_reachable")
        rpc_check = next(check for check in report.checks if check.id == "rpc_ping")
        assert tcp_check.status == "ok", "the TCP connect did succeed, so it must not be reported as failed"
        assert rpc_check.status == "fail"
        assert "TCP accepted" in rpc_check.detail
        assert "no simulator answered behind it" in rpc_check.detail
        assert "Start the Colosseum server on the remote host" in rpc_check.remedy
        assert not report.ok
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_live_readiness_gate_cannot_be_passed_by_the_fixture_fake(
    fake: tuple[EndpointConfig, FixtureFakeSimulator], tmp_path: Path
) -> None:
    endpoint, _ = fake
    config = AppConfig(endpoint=endpoint, run_class="fixture")
    gate = run_live_readiness_gate(config, PROTOCOL)
    assert gate.passed is False
    assert FIXTURE_GATE_REASON in gate.reason
    assert gate.provenance == "fixture_fake"
    failed = [check.id for check in gate.checks if check.status == "fail"]
    assert failed == ["provenance"]
    # The gate stops at provenance: it never reaches the motion or camera evidence, so a fixture that
    # moves and renders correctly still cannot pass it.
    assert [check.id for check in gate.checks] == ["connect", "provenance"]
    written = gate.write_json(tmp_path / "gate.json")
    assert json.loads(written.read_text())["passed"] is False


def test_the_fixture_does_satisfy_the_non_provenance_gate_conditions(
    adapter: ColosseumAdapter,
) -> None:
    """Pins that the gate's refusal above is about provenance alone, not a broken fixture."""
    _flying(adapter, 4.0)
    start = adapter.sample_state().position
    positions = [start]
    for _ in range(6):
        adapter.move_to(Vec3(x=start.x + 6.0, y=start.y, z=start.z), 3.0, 2.0)
        adapter.step(0.5)
        positions.append(adapter.sample_state().position)
    assert len(positions) >= 3
    assert max(start.distance_to(p) for p in positions) > 0.5
    frames = adapter.capture(("rgb", "depth"))
    assert frames["rgb"].ref.is_nonempty and frames["depth"].ref.is_nonempty


def test_build_adapter_refuses_the_fixture_when_a_live_simulator_is_required(
    fake: tuple[EndpointConfig, FixtureFakeSimulator],
) -> None:
    endpoint, _ = fake
    config = AppConfig(endpoint=endpoint, run_class="fixture", require_live_simulator=True)
    with pytest.raises(AdapterError, match="requires a live Colosseum"):
        build_adapter(config, PROTOCOL)


def test_build_adapter_returns_a_connected_adapter_for_fixture_runs(
    fake: tuple[EndpointConfig, FixtureFakeSimulator],
) -> None:
    endpoint, _ = fake
    config = AppConfig(endpoint=endpoint, run_class="fixture")
    client = build_adapter(config, PROTOCOL)
    try:
        assert client.identity().provenance == "fixture_fake"
        assert client.ping() is True
    finally:
        client.close()
