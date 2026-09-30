"""Simulator adapters: the genuine Colosseum client and the local fixture fake.

``ColosseumAdapter`` is the only adapter class. It speaks the genuine Colosseum msgpack-RPC surface and
records where its data came from. The fixture fake is a *server*, not a second adapter, so the adapter
under test in unit tests is exactly the adapter used against a live simulator.
"""

from colosseum_assurance.sim.colosseum_adapter import ColosseumAdapter, build_adapter
from colosseum_assurance.sim.diagnostics import (
    DiagnosticReport,
    GateReport,
    run_diagnostics,
    run_live_readiness_gate,
)
from colosseum_assurance.sim.fixture_fake import (
    FIXTURE_BANNER,
    FixtureFakeServer,
    FixtureFakeSimulator,
    default_obstacles,
    fixture_fake_server,
)
from colosseum_assurance.sim.identity import (
    FIXTURE_FAKE_PROBE_METHOD,
    build_identity,
    decide_provenance,
    scene_signature,
)
from colosseum_assurance.sim.scene import (
    GeometryContract,
    SceneBindResult,
    SceneBuildPlan,
    SceneMismatch,
    SceneVerificationReport,
    assert_scene_ready,
    bind_scene,
    build_plan,
    expected_actors,
    load_geometry_contract,
    verify_scene,
)

__all__ = [
    "FIXTURE_BANNER",
    "GeometryContract",
    "SceneBindResult",
    "SceneMismatch",
    "assert_scene_ready",
    "bind_scene",
    "load_geometry_contract",
    "FIXTURE_FAKE_PROBE_METHOD",
    "ColosseumAdapter",
    "DiagnosticReport",
    "FixtureFakeServer",
    "FixtureFakeSimulator",
    "GateReport",
    "SceneBuildPlan",
    "SceneVerificationReport",
    "build_adapter",
    "build_identity",
    "build_plan",
    "decide_provenance",
    "default_obstacles",
    "expected_actors",
    "fixture_fake_server",
    "run_diagnostics",
    "run_live_readiness_gate",
    "scene_signature",
    "verify_scene",
]
