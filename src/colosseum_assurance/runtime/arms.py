"""Arm assembly: one fixed controller, plus the guard that distinguishes the arm.

The controller instance is built the same way for every arm. Only ``monitor`` differs, so any behavioural
difference between arms must come from the guard, not from a different pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from colosseum_assurance.interfaces import Controller, MissionBrief, Monitor
from colosseum_assurance.protocol.spec import ArmSpec, ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import Vec3


@dataclass(slots=True)
class Arm:
    """A ready-to-run arm: its spec, its controller, and its optional guard."""

    spec: ArmSpec
    controller: Controller
    monitor: Monitor | None

    @property
    def arm_id(self) -> str:
        return self.spec.arm_id

    def describe(self) -> dict[str, Any]:
        return {
            "arm_id": self.spec.arm_id,
            "monitor_id": self.spec.monitor_id,
            "guard_enabled": self.spec.guard_enabled,
            "controller_id": getattr(self.controller, "controller_id", "unknown"),
            "monitor_description": self.monitor.describe() if self.monitor is not None else None,
        }


def build_arm(protocol: ProtocolConfig, arm_id: str) -> Arm:
    """Instantiate the controller and, for guarded arms, the monitor named in the protocol."""
    from colosseum_assurance.control.controller import InspectionController
    from colosseum_assurance.monitors import build_monitor

    spec = protocol.arms.get(arm_id)
    controller = InspectionController(protocol)
    if protocol.study_extension is not None and protocol.study_extension.policy == "finite_proxy_optimizer":
        from colosseum_assurance.control.proxy_optimizer import ProxyOptimizingController

        controller = ProxyOptimizingController(protocol)
    monitor = None
    if spec.guard_enabled:
        if spec.monitor_id is None:
            raise ValueError(f"arm {arm_id!r} enables a guard but names no monitor_id")
        monitor = build_monitor(spec.monitor_id, protocol)
        ext = protocol.study_extension
        if ext is not None and ext.perception_guard_threshold is not None:
            from colosseum_assurance.monitors.perception_guard import PerceptionConfidenceGuard

            monitor = PerceptionConfidenceGuard(monitor, ext.perception_guard_threshold,
                                                max_age_s=ext.perception_max_age_s)
    if protocol.context_confidence is not None:
        controller.enable_capture_acknowledgments(protocol.context_confidence.max_capture_opportunities)
        if arm_id in protocol.context_confidence.applies_to_arms:
            from colosseum_assurance.monitors.context_confidence import ContextConfidenceGuard

            assert monitor is not None  # validated by the v4 protocol contract
            monitor = ContextConfidenceGuard(monitor, protocol.context_confidence)
    return Arm(spec=spec, controller=controller, monitor=monitor)


def mission_brief(
    protocol: ProtocolConfig,
    manifest: ScenarioManifest,
    briefed_asset_position: Vec3 | None = None,
    brief_source: str = "protocol_nominal",
) -> MissionBrief:
    """Assemble the non-privileged brief handed to the controller and its guard.

    Only *declared* delay bounds are included. The scenario's realised delays, the jittered asset
    position, and the obstacle list stay out: perception and the guards must work without them.

    ``briefed_asset_position`` exists for the qualified-map path. When the scene binding replaces the
    generated geometry with measured bodies from a third-party level, the operator would brief the
    vehicle with the real asset location, so the brief must come from the manifest that was actually
    flown. Passing the requested manifest's nominal position there would leave the controller, its
    guard, and the evaluator reasoning about different assets. ``brief_source`` records which case
    applied, and the runner copies it into the episode record.
    """
    obs_level = next(
        level for level in protocol.conditions.observation_delay_levels
        if level.level_id == manifest.observation_delay_level
    )
    sup_level = next(
        level for level in protocol.conditions.supervision_delay_levels
        if level.level_id == manifest.supervision_delay_level
    )
    return MissionBrief(
        mission=protocol.mission,
        obligations=protocol.obligations,
        home=protocol.mission.home,
        nominal_asset_position=briefed_asset_position or protocol.mission.asset_nominal_position,
        declared_observation_delay_s=obs_level.value_s + obs_level.jitter_s,
        declared_supervision_delay_s=sup_level.value_s + sup_level.jitter_s,
        policy_version=protocol.obligations.policy_version,
        camera_name=protocol.simulation.camera_name,
        extras={
            "observation_delay_level": manifest.observation_delay_level,
            "supervision_delay_level": manifest.supervision_delay_level,
            "declared_visibility": manifest.visibility,
            "brief_source": brief_source,
            "briefed_from_scenario_id": manifest.scenario_id,
        },
    )
