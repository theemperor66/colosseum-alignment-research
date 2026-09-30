"""The closed-loop episode runner.

One episode is: reset the simulator, place the vehicle, acquire control, take off, then repeat a fixed
control step until a terminal condition. Each control step

1. samples the measured onboard state and (on the capture schedule) RGB/depth frames,
2. turns the depth frame into perception features,
3. delivers a *delayed* observation packet built from the precomputed schedule,
4. asks the fixed controller for one bounded command,
5. asks the arm's guard for a verdict and possibly an intervention,
6. executes the (possibly overridden) command,
7. advances simulator time in small sub-steps while recording dense privileged truth samples.

The runner is the only component that touches both channels. It writes the exposed record and the
privileged ledger to separate files and never passes truth into the controller or the guard.
"""

from __future__ import annotations

import copy
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colosseum_assurance.config import AppConfig
from colosseum_assurance.interfaces import (
    AdapterError,
    AdapterTimeout,
    CapturedFrame,
    SimAdapter,
)
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.runtime.arms import Arm, build_arm, mission_brief
from colosseum_assurance.runtime.evidence import EvidenceWriter, assert_provenance_allowed, utc_now
from colosseum_assurance.runtime.observation import ObservationPipeline
from colosseum_assurance.runtime.supervision import AuthorizationBroker, SupervisionLink
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import (
    AttemptedRun,
    ControlCommand,
    EpisodeRecord,
    FrameRef,
    MonitorReport,
    PrivilegedLedger,
    StepRecord,
    TerminationRecord,
    TruthEvent,
    TruthSample,
    Vec3,
    VehicleState,
    Verdict,
    sampling_quality,
)
from colosseum_assurance.sim.scene import SceneError
from colosseum_assurance.version import code_version


class SetupIncomplete(AdapterError):
    """Raised when required episode setup did not reach a verified state.

    A takeoff that never reaches cruise altitude, or a scene that cannot be verified, must terminate the
    episode as INCOMPLETE. Continuing would produce a record that looks like an ordinary research
    episode while the vehicle never entered the flight regime the study is about.
    """


@dataclass(slots=True)
class ScenarioBinding:
    """The geometry a matched arm set actually flies, established once per scenario.

    The review found the previous ordering wrong: the runner built the mission brief from the REQUESTED
    manifest, then bound the scene, then swapped only the manifest. If binding changed the asset
    location, the controller and its guard were briefed about one world while the evaluator scored
    another. Binding now happens after a reset, once per scenario, and everything downstream -- brief,
    records, ledger, evaluator -- uses the effective manifest this object carries.
    """

    requested: ScenarioManifest
    effective: ScenarioManifest
    scene_report: dict[str, Any]
    briefed_asset_position: Vec3
    brief_source: str
    bound_at_wall_clock: str

    @property
    def changed(self) -> bool:
        return self.effective.content_hash() != self.requested.content_hash()

    def deviation_record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.requested.scenario_id,
            "requested_manifest_hash": self.requested.content_hash(),
            "flown_manifest_hash": self.effective.content_hash(),
            "scene_mode": self.scene_report.get("mode"),
            "geometry_provenance": self.scene_report.get("geometry_provenance"),
            "measurement_complete": self.scene_report.get("measurement_complete"),
            "inventory_complete": self.scene_report.get("inventory_complete"),
            "blocking_reasons": self.scene_report.get("blocking_reasons", []),
            "brief_source": self.brief_source,
            "bound_at_wall_clock": self.bound_at_wall_clock,
        }


@dataclass(slots=True)
class EpisodeResult:
    """Everything one attempt produced, whether or not it finished."""

    attempt: AttemptedRun
    record: EpisodeRecord | None = None
    ledger: PrivilegedLedger | None = None
    episode_path: str | None = None
    ledger_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.attempt.status == "completed"


@dataclass
class _GuardState:
    """Latched guard effects. The guard chooses; the runner executes and records.

    ``hold_step`` exists because an independent review found that a ``hold`` intervention was logged but
    never enacted: the controller's movement still ran. A logged intervention that does not change the
    executed command would make the guarded arms look identical to the unguarded arm for that step while
    the record claimed otherwise.
    """

    suspend_inspection: bool = False
    return_to_launch: bool = False
    abort: bool = False
    hold_step: int | None = None
    interventions: list[dict[str, Any]] = field(default_factory=list)

    def hint(self) -> str | None:
        if self.abort:
            return "aborting_by_guard"
        if self.return_to_launch:
            return "returning_by_guard"
        if self.suspend_inspection:
            return "inspection_suspended_by_guard"
        return None


class EpisodeRunner:
    """Runs episodes against any :class:`SimAdapter`, fixture or live."""

    def __init__(
        self,
        adapter: SimAdapter,
        protocol: ProtocolConfig,
        config: AppConfig,
        writer: EvidenceWriter | None = None,
        save_frames: bool = True,
    ) -> None:
        self.adapter = adapter
        self.protocol = protocol
        self.config = config
        self.writer = writer
        self.save_frames = save_frames
        # Episode-relative simulator clock. A live Colosseum does not restart its clock at zero on reset,
        # so every recorded timestamp is rebased onto t = 0 at the first moment after a successful reset.
        # One clock, one origin, for records, ledger, monitors, and evaluator (docs/timing-semantics.md).
        self._time_origin_s: float = 0.0
        self._common_window: dict[str, Any] | None = None
        self._camera_obscuration: Any = None
        self._camera_perturbations: Any = None
        self._camera_episode_id: str | None = None
        self._camera_arm_id: str | None = None

    # ------------------------------------------------------------------ public
    # ------------------------------------------------------------------ binding
    def bind_scenario(self, manifest: ScenarioManifest) -> ScenarioBinding:
        """Reset, then bind and verify the scene, and return the geometry the arms will fly.

        Order matters. A reset can change or clear what is in the level, so the scene is inspected
        AFTER the reset, never before. The result is established once and reused by every arm of the
        scenario, so a matched set cannot end up flying two different worlds.
        """
        self.adapter.reset()
        self.adapter.wait_until_ready()
        reapply = getattr(self.adapter, "reapply_environment", None)
        if callable(reapply):
            reapply(manifest)

        report: dict[str, Any] = {}
        configure = getattr(self.adapter, "configure_scene", None)
        if callable(configure):
            report = dict(configure(manifest))

        effective = manifest
        payload = report.get("effective_manifest")
        if payload:
            effective = (
                payload if isinstance(payload, ScenarioManifest)
                else ScenarioManifest.model_validate(payload)
            )
        elif report.get("scenario_definition_changed"):
            bound = getattr(self.adapter, "bound_manifest", None)
            if bound is None:
                raise AdapterError(
                    "the scene binding reported a changed scenario definition but produced no effective "
                    "manifest, so the geometry that would be flown is unknown",
                    remedy="Fix the adapter's configure_scene to return `effective_manifest`.",
                )
            effective = bound

        briefed, brief_source = self._asset_brief(manifest, effective, report)
        return ScenarioBinding(
            requested=manifest,
            effective=effective,
            scene_report=report,
            briefed_asset_position=briefed,
            brief_source=brief_source,
            bound_at_wall_clock=utc_now(),
        )

    def _asset_brief(self, requested: ScenarioManifest, effective: ScenarioManifest,
                     report: dict[str, Any]) -> tuple[Vec3, str]:
        # Support, actor-name or start-pose changes do not authorize revealing the generated target
        # jitter. Only an explicit qualified-map asset or a genuinely rebound target gets a new brief.
        rebound = (report.get("mode") == "qualified_map"
                   or effective.asset_position != requested.asset_position)
        return ((effective.asset_position, "bound_scene_asset") if rebound
                else (self.protocol.mission.asset_nominal_position, "protocol_nominal"))

    def run(
        self,
        manifest: ScenarioManifest,
        arm_id: str,
        episode_id: str | None = None,
        arm: Arm | None = None,
        binding: ScenarioBinding | None = None,
    ) -> EpisodeResult:
        """Run one episode. Failures produce an incomplete record, never a silent drop.

        ``arm`` is an explicit seam: tests inject a stub guard to prove that an intervention is actually
        enacted. Experiment workflows always leave it None so the arm comes from the frozen protocol.

        ``binding`` carries the geometry established once for the whole matched arm set. When it is
        None the runner binds the scenario itself, which is the single-arm convenience path.
        """
        sim = self.protocol.simulation
        mission = self.protocol.mission
        self._common_window = None
        self._camera_obscuration = None
        controlled = self.protocol.controlled_study
        camera_spec = None if controlled is None else controlled.camera_obscuration
        if camera_spec is not None and (self.writer is None or not self.save_frames):
            raise ValueError("camera obscuration requires a writer and retained raw/delivered pixels")
        arm = arm or build_arm(self.protocol, arm_id)
        episode_id = episode_id or f"{manifest.scenario_id}__{arm_id}"
        self._camera_episode_id, self._camera_arm_id = episode_id, arm_id

        # Refuse a repeat BEFORE the simulator is touched. A second run of the same scenario and arm
        # would append a second attempt and overwrite the first episode's evidence.
        if self.writer is not None:
            self.writer.assert_episode_not_recorded(episode_id)

        # The brief describes the world that will actually be flown. When a binding exists it already
        # holds the effective geometry; otherwise the runner binds below and rebuilds the brief.
        if binding is not None:
            manifest = binding.effective
        brief = mission_brief(
            self.protocol,
            manifest,
            briefed_asset_position=None if binding is None else binding.briefed_asset_position,
            brief_source="protocol_nominal" if binding is None else binding.brief_source,
        )
        attempt_id = f"{episode_id}__{uuid.uuid4().hex[:8]}"
        started_wall = utc_now()
        wall_t0 = time.monotonic()

        identity = self.adapter.identity()
        assert_provenance_allowed(identity, manifest.run_class)
        if self.config.require_live_simulator and not identity.is_live:
            raise AdapterError(
                f"this run requires a live Colosseum simulator but provenance is {identity.provenance!r}",
                remedy="start a genuine Colosseum server and point COLASSURE_SIM_HOST/PORT at it",
            )

        samples: list[TruthSample] = []
        events: list[TruthEvent] = []
        steps: list[StepRecord] = []
        guard = _GuardState()
        termination: TerminationRecord | None = None
        error_type: str | None = None
        error_message: str | None = None
        status = "completed"
        scene_report: dict[str, Any] | None = None
        scene_setup_rejected = False
        stepping_mode_used = sim.stepping_mode

        link = SupervisionLink(schedules=manifest.schedules, obligations=self.protocol.obligations)
        broker = AuthorizationBroker(schedules=manifest.schedules, obligations=self.protocol.obligations)
        pipeline = ObservationPipeline(
            schedules=manifest.schedules,
            declared_delay_bound_s=brief.declared_observation_delay_s,
        )
        extension = self.protocol.study_extension
        if extension is not None and extension.supervision_mode == "operator_queue":
            from pathlib import Path

            from colosseum_assurance.runtime.operator import OperatorQueueBroker

            if not extension.operator_queue_directory:
                raise ValueError("operator_queue mode requires a frozen operator_queue_directory")
            broker = OperatorQueueBroker(manifest.schedules, self.protocol.obligations,
                                         Path(extension.operator_queue_directory), episode_id)
        perturbations = None
        if extension is not None:
            from colosseum_assurance.runtime.perturbations import PerturbationChannel

            perturbations = PerturbationChannel(extension, manifest.seed)
        self._camera_perturbations = perturbations

        try:
            # Reset FIRST, then inspect the scene the reset left behind, then brief from what will
            # actually be flown.
            self.adapter.reset()
            self.adapter.wait_until_ready()
            self._time_origin_s = float(self.adapter.sim_time_s())
            events.append(TruthEvent(sim_time_s=0.0, kind="reset_ok",
                                     detail="simulator reset and reported ready; episode clock origin set",
                                     payload={"absolute_sim_time_s": self._time_origin_s}))

            reapply = getattr(self.adapter, "reapply_environment", None)
            environment_report = {}
            if callable(reapply):
                environment_report = reapply(manifest)

            configure = getattr(self.adapter, "configure_scene", None)
            if callable(configure):
                scene_report = dict(configure(manifest))
                verified = self._effective_manifest(manifest, scene_report)
                if binding is None:
                    briefed, brief_source = self._asset_brief(manifest, verified, scene_report)
                    manifest = verified
                    brief = mission_brief(self.protocol, manifest, briefed_asset_position=briefed,
                                          brief_source=brief_source)
                elif verified.content_hash() != manifest.content_hash():
                    # The scene changed between arms of one matched set, so the comparison is broken.
                    raise AdapterError(
                        f"the scene no longer matches the geometry bound for scenario "
                        f"{manifest.scenario_id!r}: this arm would fly "
                        f"{verified.content_hash()} while its matched arms flew "
                        f"{manifest.content_hash()}",
                        remedy="Re-bind the scenario and re-run the whole matched arm set.",
                    )
            geometry_qualification = None
            if (self.protocol.controlled_study is not None
                    and self.protocol.controlled_study.require_feasible_inspection_geometry):
                from colosseum_assurance.scenario.qualification import inspection_geometry_qualification

                geometry_qualification = inspection_geometry_qualification(self.protocol, manifest)
                if not geometry_qualification["passed"]:
                    raise SceneError("effective authored inspection geometry failed: "
                                     + "; ".join(geometry_qualification["failures"]))
            events.append(TruthEvent(sim_time_s=0.0, kind="episode_start",
                                     detail=f"arm={arm_id} scenario={manifest.scenario_id}",
                                     payload={
                                         "schedule_hash": manifest.schedules.content_hash(),
                                         "flown_manifest_hash": manifest.content_hash(),
                                         "brief_source": brief.extras.get("brief_source"),
                                         "scene_mode": (scene_report or {}).get("mode"),
                                         "environment": environment_report,
                                         **({"authored_geometry_qualification": geometry_qualification}
                                            if geometry_qualification is not None else {}),
                                     }))
            self.adapter.set_start_pose(manifest.start_position, manifest.start_yaw_rad)
            start_pose_report = getattr(self.adapter, "start_pose_report", None)
            if start_pose_report is not None:
                # The episode_start event immediately above owns setup evidence. Keep the closed
                # event taxonomy unchanged instead of inventing a new outcome event for placement.
                events[-1].payload["start_pose_qualification"] = start_pose_report
            self.adapter.acquire_control()
            self.adapter.arm()
            events.append(TruthEvent(sim_time_s=self._now(), kind="api_control_acquired",
                                     detail="API control enabled and vehicle armed"))

            brief.extras["camera_hfov_rad"] = (getattr(self.adapter, "camera_hfov_rad", None)
                                               or float(np.deg2rad(sim.camera_hfov_deg)))
            arm.controller.reset(brief)
            if arm.monitor is not None:
                arm.monitor.reset(brief)

            segmentation_identity: dict[str, Any] = {"identity_verified": False}
            if extension is not None and extension.capture_segmentation:
                asset = next((o for o in manifest.obstacles if o.kind == "inspection_asset"), None)
                configure_mask = getattr(self.adapter, "configure_asset_segmentation", None)
                if asset is not None and callable(configure_mask):
                    segmentation_identity = configure_mask(asset.name, extension.asset_segmentation_id)
                events.append(TruthEvent(sim_time_s=self._now(), kind="segmentation_identity",
                                         detail="evaluator-only asset-mask identity contract",
                                         payload=segmentation_identity))

            takeoff_samples, takeoff_reached = self._run_takeoff(mission.cruise_altitude_m, events)
            samples.extend(takeoff_samples)
            if not takeoff_reached:
                # A failed required setup must not silently produce a normal research episode. The
                # episode ends here, incomplete, and stays visible in the attempted-run ledger.
                raise SetupIncomplete(
                    f"takeoff did not reach the commanded cruise altitude of "
                    f"{mission.cruise_altitude_m:.1f} m within the setup budget",
                    remedy=(
                        "Check the vehicle configuration and the simulator's motion API. An episode "
                        "that never reached cruise altitude is not a research episode."
                    ),
                )

            total_steps = int(round(mission.episode_horizon_s / mission.control_dt_s))
            last_phase: str | None = None
            self._common_window = None
            mission_terminal: TerminationRecord | None = None
            if self.protocol.controlled_study is not None:
                self._common_window = {
                    "semantics": self.protocol.controlled_study.model_dump(mode="json"),
                    "control_window_start_s": self._now(),
                    "required_control_duration_s": mission.episode_horizon_s,
                    "required_control_steps": total_steps, "mission_terminal": None,
                    "complete": False,
                }
                if camera_spec is not None:
                    from colosseum_assurance.runtime.perturbations import CameraObscurationChannel

                    self._camera_obscuration = CameraObscurationChannel(
                        camera_spec, self._common_window["control_window_start_s"])

            for k in range(total_steps):
                now = self._now()
                wall_now = time.monotonic() - wall_t0
                if wall_now > self.config.max_episode_wall_clock_s:
                    termination = TerminationRecord(
                        reason="operator_interrupt", step_index=k, sim_time_s=now,
                        detail="episode exceeded max_episode_wall_clock_s", reached_terminal_state=False,
                    )
                    status = "timeout"
                    break

                state = self._rebase_state(self.adapter.sample_state())
                if self._camera_obscuration is not None:
                    events.extend(self._camera_obscuration.update(now))
                if perturbations is not None:
                    events.extend(perturbations.update(now))
                    raw_onboard_state = state.model_dump(mode="json")
                    state = perturbations.state(state)
                    if self.protocol.controlled_study is not None and self.writer is not None:
                        self.writer.append_state_transformation(episode_id, {
                            "scenario_id": manifest.scenario_id,
                            "flown_manifest_hash": manifest.content_hash(),
                            "arm_id": arm_id, "step_index": k, "now_s": now,
                            "family": extension.family, "severity": extension.severity,
                            "active": perturbations.active, "fault_onset_s": extension.fault_onset_s,
                            "fault_duration_s": extension.fault_duration_s,
                            "raw_state": raw_onboard_state, "delivered_state": state.model_dump(mode="json"),
                            "semantics": "same onboard acquisition pre/post injection; before delay/dropout",
                        })
                pipeline.push_state(state)
                if self.protocol.controlled_study is not None:
                    mode_reader = getattr(self.adapter, "control_mode_sample", None)
                    mode = (mode_reader() if callable(mode_reader)
                            else {"available": False, "error": "adapter lacks onboard mode evidence"})
                    if mode.get("available"):
                        mode["sample"]["sim_time_s"] -= self._time_origin_s
                    pipeline.push_sensors({"actuation_mode": mode})
                if extension is not None and extension.capture_sensors:
                    sample_sensors = getattr(self.adapter, "sample_sensors", None)
                    sensor_samples = sample_sensors() if callable(sample_sensors) else {
                        "imu": {"available": False, "error": "adapter has no sensor interface"},
                        "gps": {"available": False, "error": "adapter has no sensor interface"},
                    }
                    for value in sensor_samples.values():
                        if value.get("available"):
                            value["sample"]["sim_time_s"] -= self._time_origin_s
                    pipeline.push_sensors(sensor_samples)

                frames: dict[str, CapturedFrame] = {}
                perception_prediction: dict[str, Any] | None = None
                if k % sim.capture_every_n_steps == 0 and (sim.capture_rgb or sim.capture_depth):
                    wanted = (("rgb", sim.capture_rgb), ("depth", sim.capture_depth))
                    kinds = tuple(x for x, on in wanted if on)
                    if extension is not None and extension.capture_segmentation:
                        kinds += ("segmentation",)
                    prefix = None
                    save_now = (self._camera_obscuration is not None
                                or self.save_frames and k % sim.save_frames_every_n_steps == 0)
                    if save_now and self.writer is not None:
                        prefix = str(self.writer.frame_prefix(episode_id, k).resolve())
                    try:
                        frames = {
                            name: self._rebase_frame(frame)
                            for name, frame in self.adapter.capture(kinds=kinds, save_prefix=prefix).items()
                        }
                    except AdapterError:
                        if (extension is not None and extension.capture_segmentation
                                and self.writer is not None):
                            from colosseum_assurance.runtime.perception_capture import record_perception_frame

                            record_perception_frame(
                                frames={}, spec=extension, identity=segmentation_identity,
                                scenario_group=f"{manifest.run_class}:environment-{manifest.seed}",
                                arm_id=arm_id, frame_id=f"{episode_id}:step-{k}",
                                provenance=identity.provenance,
                                stratum=f"{manifest.layout_variant}/{manifest.visibility}",
                                out=self.writer.root / "perception" / "rows.jsonl", dropped=True,
                                hfov_rad=brief.extras["camera_hfov_rad"],
                            )
                        raise
                    if self._camera_obscuration is not None:
                        frames = self._obscure_camera_acquisition(
                            frames, prefix, k, now, manifest, "observation")
                    elif perturbations is not None:
                        frames = perturbations.frames(frames)
                        if prefix is not None:
                            # Preserve raw and delivered pixels separately. Their metadata must not
                            # point at different pixels than those used by the vision controller.
                            for name, frame in frames.items():
                                if frame.array is not None:
                                    path = prefix + f"_delivered_{name}.npy"
                                    np.save(path, frame.array)
                                    frame.ref = frame.ref.model_copy(
                                        update={"path": path, "pixels_as": "npy"})
                    if extension is not None and extension.capture_segmentation and self.writer is not None:
                        from colosseum_assurance.runtime.perception_capture import record_perception_frame

                        perception_prediction = record_perception_frame(
                            frames=frames, spec=extension, identity=segmentation_identity,
                            scenario_group=f"{manifest.run_class}:environment-{manifest.seed}", arm_id=arm_id,
                            frame_id=f"{episode_id}:step-{k}", provenance=identity.provenance,
                            stratum=f"{manifest.layout_variant}/{manifest.visibility}",
                            out=self.writer.root / "perception" / "rows.jsonl",
                            dropped=manifest.schedules.depth_dropped(k),
                            hfov_rad=brief.extras["camera_hfov_rad"],
                            context_policy=self.protocol.context_confidence,
                        )
                        pipeline.push_prediction(perception_prediction)
                    depth_frame = frames.get("depth")
                    rgb_frame = frames.get("rgb")
                    summary = self._summarize(depth_frame, rgb_frame, now, manifest, onboard_state=state)
                    if summary is not None:
                        pipeline.push_depth(summary)
                    if rgb_frame is not None and rgb_frame.ref.acquisition_time_known:
                        # A frame of unknown age is not usable evidence, so it never reaches the vehicle.
                        pipeline.push_rgb(rgb_frame.ref)

                events.extend(link.advance_to(now))
                events.extend(broker.advance_to(now))
                observation = pipeline.build(
                    step_index=k,
                    now_s=now,
                    supervision=link.view(now),
                    authorization=broker.view(now),
                    mission_phase_hint=guard.hint(),
                )

                following_up = self._common_window is not None and mission_terminal is not None
                if following_up:
                    command = ControlCommand(
                        step_index=k, issued_sim_time_s=now, kind="hold", duration_s=mission.control_dt_s,
                        issued_by="runner", controller_phase="common_followup",
                        reason="frozen terminal policy: hold position or remain acknowledged disarmed",
                    )
                    controller_state = {"phase": "common_followup",
                                        "mission_terminal": mission_terminal.model_dump(mode="json")}
                else:
                    command = arm.controller.step(observation)
                    controller_state = dict(arm.controller.internal_state())
                phase = str(controller_state.get("phase", "unknown"))
                terminal_policy_active = (following_up or (self._common_window is not None
                                                          and phase in {"done", "aborted"}))
                if phase != last_phase:
                    events.append(TruthEvent(sim_time_s=now, kind="controller_phase_change",
                                             detail=phase, payload={"claimed_by": "controller"}))
                    last_phase = phase

                report: MonitorReport | None = None
                if arm.monitor is not None:
                    report = arm.monitor.evaluate(observation, command)
                    if not terminal_policy_active:
                        self._apply_intervention(report, guard, now, k, events)

                execution_command = command
                if terminal_policy_active and not following_up:
                    # The terminal policy begins at the terminal decision itself. A previously
                    # latched RTL must never turn the controller's done/noop into powered motion.
                    execution_command = ControlCommand(
                        step_index=k, issued_sim_time_s=now, kind="hold", duration_s=mission.control_dt_s,
                        issued_by="runner", reason="terminal hold or remain acknowledged disarmed",
                    )
                active_guard = _GuardState() if terminal_policy_active else guard
                executed = self._execute(execution_command, active_guard,
                                         broker, manifest, now, k, events, frames)
                if self.protocol.context_confidence is not None and command.kind == "inspect_capture":
                    image = frames.get("rgb")
                    rgb_acquired = bool(image is not None and image.array is not None and image.array.size
                                        and image.ref.width > 0 and image.ref.height > 0
                                        and image.ref.acquisition_time_known)
                    feedback = arm.controller.acknowledge_capture(command, executed, rgb_acquired)
                    controller_state = dict(arm.controller.internal_state())
                    observed_permission = (None if report is None else report.context_confidence_evidence)
                    for event in events:
                        if (event.kind == "inspection_capture_performed"
                                and event.payload.get("step_index") == k):
                            event.payload["context_confidence_permission"] = observed_permission
                            event.payload["capture_execution_feedback"] = feedback
                            event.payload["capture_identity_scope"] = (
                                "Executed frame refs are separate from observed-frame permission; "
                                "permission does not certify newer-image content.")
                before_advance = self._now()
                if self._common_window is None:
                    step_samples = self._advance(mission.control_dt_s, events)
                else:
                    # Bound each step to its absolute control-window deadline, so small native
                    # under-advances cannot accumulate into a shorter arm-specific safety window.
                    target = self._common_window["control_window_start_s"] + (k+1)*mission.control_dt_s
                    step_samples = []
                    for _ in range(5):
                        remaining = target-self._now()
                        if remaining <= 1e-6:
                            break
                        step_samples.extend(self._advance(remaining, events))
                    if self._now() < target-1e-6:
                        samples.extend(step_samples)
                        raise AdapterError("common follow-up clock did not reach its bounded step deadline")
                samples.extend(step_samples)
                sim_advance_s = round(self._now() - before_advance, 6)

                steps.append(
                    StepRecord(
                        step_index=k,
                        sim_time_s=now,
                        wall_clock_s=round(wall_now, 4),
                        observation=observation,
                        command=command,
                        executed_command=executed,
                        monitor_report=report,
                        controller_state=controller_state,
                        provenance={
                            "policy_version": self.protocol.obligations.policy_version,
                            "protocol_hash": manifest.protocol_hash,
                            "scenario_id": manifest.scenario_id,
                            "flown_manifest_hash": manifest.content_hash(),
                            "requested_manifest_hash": (
                                binding.requested.content_hash() if binding is not None
                                else manifest.content_hash()
                            ),
                            "brief_source": brief.extras.get("brief_source"),
                            "arm_id": arm_id,
                            "monitor_id": arm.spec.monitor_id,
                            "observation_delay_level": manifest.observation_delay_level,
                            "supervision_delay_level": manifest.supervision_delay_level,
                            "stepping_mode": stepping_mode_used,
                            # Recorded so a regression test can prove exactly one dt was consumed per
                            # control step and that the interval was densely sampled.
                            "sim_advance_s": sim_advance_s,
                            "truth_samples_this_step": len(step_samples),
                            "frames_captured": sorted(frames),
                            **({"controlled_followup": terminal_policy_active,
                                "terminal_monitor_recommendations_applied": not terminal_policy_active}
                               if self._common_window is not None else {}),
                            "perception_prediction": perception_prediction,
                            "guard_state": {
                                "suspend_inspection": guard.suspend_inspection,
                                "return_to_launch": guard.return_to_launch,
                                "abort": guard.abort,
                            },
                        },
                    )
                )

                candidate = None
                if guard.abort:
                    candidate = TerminationRecord(
                        reason="guard_abort", step_index=k, sim_time_s=self._now(),
                        detail="guard requested abort", completed_mission=False,
                    )
                    if self._common_window is None:
                        status = "aborted"
                elif phase == "done":
                    candidate = TerminationRecord(
                        reason="mission_complete", step_index=k, sim_time_s=self._now(),
                        detail="controller declared the mission finished", completed_mission=True,
                    )
                elif phase == "aborted":
                    candidate = TerminationRecord(
                        reason="controller_abort", step_index=k, sim_time_s=self._now(),
                        detail=str(controller_state.get("reason", "controller aborted")),
                    )
                elif guard.return_to_launch and self._returned_home(samples, manifest):
                    candidate = TerminationRecord(
                        reason="guard_return_complete", step_index=k, sim_time_s=self._now(),
                        detail="guard-commanded return reached home",
                    )
                if candidate is not None:
                    if self._common_window is None:
                        termination = candidate
                        break
                    if mission_terminal is None:
                        mission_terminal = candidate
                        self._common_window["mission_terminal"] = candidate.model_dump(mode="json")
                        events.append(TruthEvent(
                            sim_time_s=self._now(), kind="controller_phase_change",
                            detail="mission ending recorded; common follow-up begins",
                            payload={"controlled_study_mission_terminal": candidate.model_dump(mode="json")},
                        ))
            else:
                termination = TerminationRecord(
                    reason="horizon_reached", step_index=total_steps - 1,
                    sim_time_s=self._now(),
                    detail=f"episode horizon {mission.episode_horizon_s} s elapsed",
                )
                if self._common_window is not None:
                    self._common_window["complete"] = True
                    self._common_window["control_window_end_s"] = self._now()
                    self._common_window["observed_control_duration_s"] = (
                        self._now()-self._common_window["control_window_start_s"])

        except (SetupIncomplete, SceneError) as exc:
            scene_setup_rejected = isinstance(exc, SceneError)
            error_type, error_message, status = type(exc).__name__, str(exc), "partial"
            termination = TerminationRecord(
                reason="setup_failed", step_index=len(steps), sim_time_s=self._safe_time(),
                detail=str(exc), reached_terminal_state=False,
            )
            events.append(TruthEvent(sim_time_s=self._safe_time(), kind="simulator_error",
                                     detail=f"setup incomplete: {exc}"))
        except AdapterTimeout as exc:
            error_type, error_message, status = type(exc).__name__, str(exc), "timeout"
            termination = TerminationRecord(
                reason="rpc_timeout", step_index=len(steps), sim_time_s=self._safe_time(),
                detail=str(exc), reached_terminal_state=False,
            )
            events.append(TruthEvent(sim_time_s=self._safe_time(), kind="simulator_error", detail=str(exc)))
        except AdapterError as exc:
            error_type, error_message, status = type(exc).__name__, str(exc), "crashed"
            reason = "reset_failed" if "reset" in str(exc).lower() else "simulator_error"
            termination = TerminationRecord(
                reason=reason, step_index=len(steps), sim_time_s=self._safe_time(),
                detail=str(exc), reached_terminal_state=False,
            )
            events.append(TruthEvent(sim_time_s=self._safe_time(), kind="simulator_error", detail=str(exc)))
        except Exception as exc:  # noqa: BLE001 - a runner bug must stay visible in the ledger
            error_type, error_message, status = type(exc).__name__, str(exc), "crashed"
            termination = TerminationRecord(
                reason="runner_exception", step_index=len(steps), sim_time_s=self._safe_time(),
                detail=traceback.format_exc(limit=4), reached_terminal_state=False,
            )
        finally:
            self._safe_teardown(events)

        assert termination is not None
        sim_duration = self._safe_time()
        end_event = TruthEvent(sim_time_s=sim_duration, kind="episode_end",
                               detail=termination.reason,
                               payload={"completed_mission": termination.completed_mission})
        if self._common_window is not None:
            end_event.payload["controlled_study_followup"] = copy.deepcopy(self._common_window)
        landing_report = getattr(self.adapter, "landing_report", None)
        if landing_report is not None:
            end_event.payload["native_landing_handshake"] = {
                "episode_id": episode_id, "report": copy.deepcopy(landing_report),
            }
        events.append(end_event)

        # A terminal truth sample makes the end of the episode observable to the evaluator instead of
        # leaving a blind interval between the last control step and termination.
        if not scene_setup_rejected:
            try:
                samples.append(self._rebase_truth(self.adapter.sample_truth()))
            except Exception as exc:  # noqa: BLE001 - a missing terminal sample is recorded, not hidden
                events.append(TruthEvent(sim_time_s=sim_duration, kind="simulator_error",
                                         detail=f"terminal truth sample unavailable: {exc}"))
        # A rejected pre-flight scene has no flown truth interval. Keep an empty, zero-coverage
        # ledger rather than adding one t=0 diagnostic point that cannot establish coverage and
        # would prevent the setup-failure attempt from being persisted by the strict ledger schema.

        expected_samples = max(0, int(round(sim_duration / sim.truth_sample_interval_s)))
        if expected_samples == 0 and samples:
            # An immediate setup/RPC failure can yield one terminal diagnostic point but no expected
            # sampling interval. Keep those actual points verbatim as privileged diagnostics; do not
            # invent a positive denominator/coverage or let strict ledger validation erase the failed
            # attempt. The primary flight ledger remains empty and its physical verdicts unknown.
            end_event.payload["zero_interval_diagnostic_truth_samples"] = [
                sample.model_dump(mode="json") for sample in samples
            ]
            end_event.payload["diagnostic_truth_scope"] = (
                "Measured points retained outside flight-sampling coverage because expected_sample_count=0"
            )
            samples = []
        coverage, max_gap, achieved_interval = sampling_quality(samples, expected_samples)

        # Measured clock facts from the adapter, so a degraded or fallback stepping mode is visible in
        # the evidence rather than inferred from the requested configuration.
        timing_report: dict[str, Any] = {}
        reporter = getattr(self.adapter, "stepping_report", None)
        if callable(reporter):
            try:
                timing_report = dict(reporter())
            except Exception as exc:  # noqa: BLE001 - a reporting failure must not hide the episode
                timing_report = {"error": f"{type(exc).__name__}: {exc}"}
        if self._common_window is not None:
            timing_report["controlled_study_followup"] = copy.deepcopy(self._common_window)

        record = EpisodeRecord(
            episode_id=episode_id,
            scenario_id=manifest.scenario_id,
            arm_id=arm_id,
            run_class=manifest.run_class,
            protocol_hash=manifest.protocol_hash,
            policy_version=self.protocol.obligations.policy_version,
            code_version=code_version(),
            simulator_identity=identity,
            started_wall_clock=started_wall,
            dt_s=mission.control_dt_s,
            steps=steps,
            termination=termination,
            monitor_id=arm.spec.monitor_id,
            interventions=guard.interventions,
            timing_report=timing_report,
            notes=f"arm={arm.describe()} scene_mode={(scene_report or {}).get('mode', 'unconfigured')}",
        )
        if scene_report is not None:
            record.notes = (
                f"{record.notes} | scene_verified={scene_report.get('ok', 'unknown')} "
                f"geometry={scene_report.get('geometry_provenance', 'unknown')}"
            )
        ledger = PrivilegedLedger(
            episode_id=episode_id,
            scenario_id=manifest.scenario_id,
            arm_id=arm_id,
            run_class=manifest.run_class,
            protocol_hash=manifest.protocol_hash,
            simulator_identity=identity,
            sample_interval_s=sim.truth_sample_interval_s,
            samples=samples,
            events=sorted(events, key=lambda e: e.sim_time_s),
            termination=termination,
            truth_coverage_fraction=round(coverage, 6),
            expected_sample_count=expected_samples,
            timing_report=timing_report,
            max_sample_gap_s=None if max_gap is None else round(max_gap, 6),
            achieved_sample_interval_s=None if achieved_interval is None else round(achieved_interval, 6),
            notes=(
                "privileged: evaluator input only. Coverage, maximum gap, and achieved interval are "
                "measured from sample timestamps, not inferred from the requested interval."
            ),
        )
        if status == "completed" and not termination.reached_terminal_state:
            status = "partial"

        attempt = AttemptedRun(
            attempt_id=attempt_id,
            episode_id=episode_id,
            scenario_id=manifest.scenario_id,
            arm_id=arm_id,
            run_class=manifest.run_class,
            protocol_hash=manifest.protocol_hash,
            status=status,  # type: ignore[arg-type]
            started_wall_clock=started_wall,
            finished_wall_clock=utc_now(),
            wall_clock_duration_s=round(time.monotonic() - wall_t0, 3),
            sim_duration_s=round(sim_duration, 3),
            termination_reason=termination.reason,
            simulator_provenance=identity.provenance,
            error_type=error_type,
            error_message=error_message,
        )

        result = EpisodeResult(attempt=attempt, record=record, ledger=ledger)
        if self.writer is not None:
            self.writer.note_simulator(identity)
            # The manifest written is the one that was FLOWN, so the evaluator, the audit reference and
            # any replay all read the same geometry the vehicle was in.
            self.writer.write_manifest(manifest)
            if binding is not None and binding.changed:
                self.writer.record_scenario_deviation(
                    {"episode_id": episode_id, "arm_id": arm_id, **binding.deviation_record()}
                )
            result.episode_path = str(self.writer.write_episode(record))
            result.ledger_path = str(self.writer.write_ledger(ledger))
            attempt.episode_record_path = result.episode_path
            attempt.ledger_path = result.ledger_path
            self.writer.append_attempt(attempt)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _effective_manifest(manifest: ScenarioManifest, report: dict[str, Any]) -> ScenarioManifest:
        """The geometry the scene report says will actually be flown."""
        payload = report.get("effective_manifest")
        if payload:
            return (
                payload if isinstance(payload, ScenarioManifest)
                else ScenarioManifest.model_validate(payload)
            )
        return manifest

    def _now(self) -> float:
        """Episode-relative simulator time in seconds, clamped at zero."""
        return max(0.0, float(self.adapter.sim_time_s()) - self._time_origin_s)

    def _safe_time(self) -> float:
        try:
            return self._now()
        except Exception:  # noqa: BLE001 - never hide the original failure behind a clock failure
            return 0.0

    def _rebase_state(self, state: VehicleState) -> VehicleState:
        return state.model_copy(update={"sim_time_s": max(0.0, state.sim_time_s - self._time_origin_s)})

    def _rebase_truth(self, truth: TruthSample) -> TruthSample:
        return truth.model_copy(update={"sim_time_s": max(0.0, truth.sim_time_s - self._time_origin_s)})

    def _rebase_frame(self, frame: CapturedFrame) -> CapturedFrame:
        """Rebase a frame's acquisition time onto the episode clock, leaving an unknown time unknown."""
        ref = frame.ref
        if ref.sim_time_s is None:
            return frame
        frame.ref = ref.model_copy(
            update={"sim_time_s": max(0.0, ref.sim_time_s - self._time_origin_s)}
        )
        return frame

    def _obscure_camera_acquisition(self, raw: dict[str, CapturedFrame], prefix: str | None,
                                   step_index: int, now_s: float, manifest: ScenarioManifest,
                                   role: str) -> dict[str, CapturedFrame]:
        from pathlib import Path

        from colosseum_assurance.runtime.perception_capture import retain_camera_transformation

        assert self.writer is not None and prefix is not None and self._camera_obscuration is not None
        before = (self._camera_perturbations.frames(raw)
                  if self._camera_perturbations is not None else raw)
        delivered, report = self._camera_obscuration.frames(before, requested_s=now_s)
        acquisition_id = f"{self._camera_episode_id}:step-{step_index}"
        if role != "observation":
            acquisition_id += f":{role}"
        delivered = retain_camera_transformation(
            raw=raw, before=before, delivered=delivered, report=report, prefix=Path(prefix),
            sidecar=(self.writer.root / "privileged_camera_transformations"
                     / f"{self._camera_episode_id}.jsonl"),
            identity={"episode_id": self._camera_episode_id, "arm_id": self._camera_arm_id,
                      "scenario_id": manifest.scenario_id, "protocol_hash": self.protocol.content_hash(),
                      "flown_manifest_hash": manifest.content_hash(), "step_index": step_index,
                      "acquisition_id": acquisition_id, "role": role,
                      "legacy_photometric_seed": manifest.seed,
                      "legacy_brightness_multiplier": (.65, .85, 1., 1.15)[manifest.seed % 4],
                      "episode_clock_origin_s": self._time_origin_s},
        )
        if delivered is None:
            raise AdapterError("camera-only acquisition pairing refused: " + "; ".join(report["reasons"]))
        return delivered

    def _safe_teardown(self, events: list[TruthEvent]) -> None:
        try:
            self.adapter.release_control()
            events.append(TruthEvent(sim_time_s=self._safe_time(), kind="api_control_released",
                                     detail="control released at episode end"))
        except Exception as exc:  # noqa: BLE001
            events.append(TruthEvent(sim_time_s=self._safe_time(), kind="simulator_error",
                                     detail=f"release_control failed: {exc}"))

    def _advance(self, dt_s: float, events: list[TruthEvent]) -> list[TruthSample]:
        """Advance simulator time in truth-sample sub-steps, recording dense privileged truth."""
        interval = self.protocol.simulation.truth_sample_interval_s
        n_sub = max(1, int(round(dt_s / interval)))
        out: list[TruthSample] = []
        for _ in range(n_sub):
            self.adapter.step(interval)
            truth = self._rebase_truth(self.adapter.sample_truth())
            out.append(truth)
            if truth.collision_active:
                events.append(TruthEvent(
                    sim_time_s=truth.sim_time_s, kind="collision",
                    detail=truth.collision_object or "unnamed object",
                    payload={
                        "object": truth.collision_object,
                        "count": truth.collision_count,
                        "penetration_m": truth.collision_penetration_m,
                        "position": truth.position.model_dump(),
                        "speed_mps": float(np.linalg.norm(list(truth.velocity.as_tuple()))),
                    },
                ))
        return out

    def _run_takeoff(self, altitude_m: float, events: list[TruthEvent]) -> tuple[list[TruthSample], bool]:
        """Take off with the runner owning the clock, so takeoff is covered by dense truth samples.

        Two defects are avoided here.

        * The adapter's blocking ``takeoff`` advances time by itself and samples nothing. Using it would
          leave the whole climb invisible to the evaluator, so the runner advances the clock instead.
        * A single bounded climb command EXPIRES. Upstream ``moveToPosition`` delegates to
          ``moveOnPath``, whose loop is bounded by the supplied timeout
          (MultirotorApiBase.cpp:311-329, 428-435 at the pin). Issuing one command with a 0.5 s budget
          and then waiting 12 s means the vehicle stops climbing after the first half second. The climb
          command is therefore REFRESHED on every control step until the altitude is reached.

        Returns the truth samples collected and whether the commanded altitude was actually reached.
        """
        mission = self.protocol.mission
        budget_s = max(4.0 * mission.control_dt_s, 12.0)
        tolerance_m = 0.6
        self.adapter.issue_takeoff(timeout_s=budget_s)
        collected: list[TruthSample] = []
        elapsed = 0.0
        reached = False
        refreshes = 0
        target = Vec3(x=mission.home.x, y=mission.home.y, z=mission.home.z - abs(float(altitude_m)))
        while elapsed < budget_s:
            collected.extend(self._advance(mission.control_dt_s, events))
            elapsed += mission.control_dt_s
            if not collected:
                continue
            height = -collected[-1].position.z
            if abs(height - abs(float(altitude_m))) <= tolerance_m:
                reached = True
                break
            if height > 0.5:
                # Refresh the bounded climb every step so it cannot silently expire mid-climb.
                self.adapter.move_to(target, mission.cruise_speed_mps, mission.control_dt_s)
                refreshes += 1
        final_height = -collected[-1].position.z if collected else 0.0
        events.append(TruthEvent(
            sim_time_s=self._safe_time(),
            kind="takeoff_complete" if reached else "simulator_error",
            detail=(
                f"climb to {altitude_m:.1f} m reached within {tolerance_m:.1f} m"
                if reached else
                f"climb did not reach {altitude_m:.1f} m within {budget_s:.1f} s "
                f"(last height {final_height:.2f} m)"
            ),
            payload={
                "commanded_altitude_m": float(altitude_m),
                "final_height_m": round(final_height, 3),
                "tolerance_m": tolerance_m,
                "budget_s": budget_s,
                "climb_command_refreshes": refreshes,
                "reached": reached,
            },
        ))
        return collected, reached

    def _summarize(
        self,
        depth: CapturedFrame | None,
        rgb: CapturedFrame | None,
        now_s: float,
        manifest: ScenarioManifest,
        onboard_state: VehicleState | None = None,
    ):
        """Run the perception path on the captured frames (never on privileged geometry).

        The summary is stamped with the frame's ACQUISITION time, not with the moment it was processed.
        Stamping it with ``now`` erased the camera's native staleness: a frame acquired at t=2 and
        processed at t=10 looked zero seconds old, so the guard saw only the injected delay. The
        observation pipeline then adds the scheduled delay on top of the real age, which is the
        quantity the study is about.

        A frame whose acquisition time is unknown returns None. It is not evidence: nothing can be said
        about how stale it is, and dating it to now would present arbitrarily old pixels as fresh.
        """
        from colosseum_assurance.control.perception import summarize_depth
        from colosseum_assurance.schemas import DepthSummary

        if depth is None:
            return None
        ref: FrameRef | None = depth.ref
        if ref is None or not ref.acquisition_time_known or ref.sim_time_s is None:
            return None
        acquired_s = min(float(ref.sim_time_s), now_s)

        hfov_rad = float(np.deg2rad(self.protocol.simulation.camera_hfov_deg))
        degraded = None if manifest.visibility == "clear" else f"visibility={manifest.visibility}"
        rgb_array = None
        if rgb is not None and rgb.ref is not None and rgb.ref.acquisition_time_known:
            rgb_array = rgb.array
        camera_mount = getattr(self.adapter, "camera_mount", None)
        if (hasattr(self.adapter, "camera_mount") and not self.adapter.is_fixture_fake
                and camera_mount is None):
            raise AdapterError("live perception requires a verified fixed camera mounting calibration")
        projection = {}
        if camera_mount is not None:
            if onboard_state is None or onboard_state.orientation_wxyz is None:
                raise AdapterError("live camera projection requires full ONBOARD estimated attitude")
            frame_report = getattr(self.adapter, "scene_frame_report", None) or {}
            hfov_rad = getattr(self.adapter, "camera_hfov_rad", None)
            if hfov_rad is None:
                raise AdapterError("live camera projection requires calibrated intrinsics")
            pairing_gap = abs(float(ref.sim_time_s)-onboard_state.sim_time_s)
            if pairing_gap > self.config.camera_max_attitude_pairing_gap_s:
                # An old rendered frame cannot be projected with a new attitude. Keep its original
                # acquisition time and explicitly degrade, without consulting absolute camera truth.
                return DepthSummary(
                    sim_time_s=acquired_s, camera_name=self.protocol.simulation.camera_name,
                    valid=False, frame=ref, degraded_reason="camera_attitude_timestamp_mismatch",
                    geometry_source="calibrated_projection_unavailable_unpaired_onboard_attitude",
                    projection_attitude_sim_time_s=onboard_state.sim_time_s,
                    camera_mount_evidence_sha256=frame_report.get(
                        "camera_mount_calibration", {}).get("sha256"),
                )
            projection = dict(
                camera_mount=camera_mount, onboard_orientation_wxyz=onboard_state.orientation_wxyz,
                onboard_yaw_rad=onboard_state.yaw_rad, attitude_sim_time_s=onboard_state.sim_time_s,
                mount_evidence_sha256=frame_report.get("camera_mount_calibration", {}).get("sha256"),
            )
        return summarize_depth(
            depth.array,
            sim_time_s=acquired_s,
            camera_name=self.protocol.simulation.camera_name,
            hfov_rad=hfov_rad,
            frame_ref=ref,
            rgb=rgb_array,
            degraded_reason=degraded,
            **projection,
        )

    def _apply_intervention(
        self,
        report: MonitorReport,
        guard: _GuardState,
        now_s: float,
        step_index: int,
        events: list[TruthEvent],
    ) -> None:
        """Latch the guard effect the monitor asked for and record it in both channels."""
        if report.intervention == "none":
            return
        if report.intervention == "hold":
            # Enact it: this step's executed command becomes a hold, whatever the controller asked for.
            guard.hold_step = step_index
            changed = True
        elif report.intervention == "suspend_inspection":
            changed = not guard.suspend_inspection
            guard.suspend_inspection = True
        elif report.intervention == "return_to_launch":
            changed = not guard.return_to_launch
            guard.return_to_launch = True
            guard.suspend_inspection = True
        elif report.intervention == "abort":
            changed = not guard.abort
            guard.abort = True
        else:  # pragma: no cover - the literal type makes this unreachable
            raise ValueError(f"unknown intervention {report.intervention!r}")

        entry = {
            "intervention_id": f"guard-{step_index:06d}",
            "source": "runtime_guard",
            "step_index": step_index,
            "sim_time_s": now_s,
            "intervention": report.intervention,
            "monitor_id": report.monitor_id,
            "verdict": report.verdict.value,
            "rationale": report.rationale,
            "first_of_kind": changed or report.intervention == "hold",
        }
        guard.interventions.append(entry)
        events.append(TruthEvent(sim_time_s=now_s, kind="guard_intervention",
                                 detail=f"{report.monitor_id}:{report.intervention}", payload=entry))

    def _execute(
        self,
        command: ControlCommand,
        guard: _GuardState,
        broker: AuthorizationBroker,
        manifest: ScenarioManifest,
        now_s: float,
        step_index: int,
        events: list[TruthEvent],
        frames: dict[str, CapturedFrame],
    ) -> ControlCommand:
        """Apply guard overrides, then send one bounded command to the simulator."""
        mission = self.protocol.mission
        dt = mission.control_dt_s
        executed = command

        # A hold suppresses movement for this step. A landing is exempt: interrupting a descent is not
        # the safer action, and the guard has stronger tools (return_to_launch, abort) if it wants one.
        if guard.hold_step == step_index and command.kind not in {"land", "hold"}:
            executed = ControlCommand(
                step_index=step_index, issued_sim_time_s=now_s, kind="hold", duration_s=dt,
                reason="guard hold: movement suppressed for this step", issued_by="guard",
            )
        if guard.abort:
            executed = ControlCommand(
                step_index=step_index, issued_sim_time_s=now_s, kind="hold", duration_s=dt,
                reason="guard abort: holding before teardown", issued_by="guard",
            )
        elif guard.return_to_launch and command.kind not in {"land", "hold"}:
            home = Vec3(x=mission.home.x, y=mission.home.y, z=mission.home.z - mission.cruise_altitude_m)
            executed = ControlCommand(
                step_index=step_index, issued_sim_time_s=now_s, kind="move_to", target=home,
                speed_mps=mission.cruise_speed_mps, duration_s=dt,
                reason="guard return-to-launch override", issued_by="guard",
            )
        elif guard.suspend_inspection and command.kind == "inspect_capture":
            executed = ControlCommand(
                step_index=step_index, issued_sim_time_s=now_s, kind="hold", duration_s=dt,
                reason="guard suspended the inspection step", issued_by="guard",
            )

        # Only `step` advances simulated time, and the runner calls it after this method returns. Every
        # branch below therefore ISSUES a command and returns immediately; a blocking adapter call here
        # would advance the clock a second time and leave that interval without truth samples.
        kind = executed.kind
        if kind == "request_authorization":
            events.append(broker.request(now_s))
            self.adapter.issue_hold()
        elif kind == "inspect_capture":
            if not frames and self._camera_obscuration is not None:
                assert self.writer is not None and self._camera_episode_id is not None
                prefix = str(self.writer.frame_prefix(self._camera_episode_id, step_index).resolve())
                prefix += "_execution"
                raw = {name: self._rebase_frame(frame) for name, frame in self.adapter.capture(
                    kinds=("rgb", "depth", "segmentation"), save_prefix=prefix).items()}
                captured = self._obscure_camera_acquisition(
                    raw, prefix, step_index, now_s, manifest, "execution_fallback")
                frames.update(captured)
            else:
                captured = frames or self.adapter.capture(kinds=("rgb", "depth"))
            refs = {name: cf.ref.model_dump(mode="json") for name, cf in captured.items()}
            nonempty = {name: bool(cf.ref.is_nonempty) for name, cf in captured.items()}
            truth = self.adapter.sample_truth()
            events.append(TruthEvent(
                sim_time_s=now_s, kind="inspection_capture_performed",
                detail="close-range inspection capture",
                payload={
                    "frames": refs,
                    "frames_nonempty": nonempty,
                    "true_position": truth.position.model_dump(),
                    "true_distance_to_asset_m": round(truth.position.distance_to(manifest.asset_position), 4),
                    "authorization_view_status": broker.view(now_s).status.value,
                    "token_id": broker.view(now_s).token_id,
                    "step_index": step_index,
                    "response_source": ("synthetic_broker" if self.protocol.study_extension is None
                                        else self.protocol.study_extension.supervision_mode),
                },
            ))
            self.adapter.issue_hold()
        elif kind == "move_to":
            target = executed.target
            if target is None:
                raise ValueError("move_to command without a target")
            speed = min(float(executed.speed_mps or mission.cruise_speed_mps), mission.cruise_speed_mps)
            duration = min(float(executed.duration_s or dt), self.protocol.simulation.max_command_duration_s)
            self.adapter.move_to(target, speed_mps=speed, duration_s=duration, yaw_rad=executed.yaw_rad)
        elif kind == "hold":
            # A hover does not rotate. A controller hold that carries a heading is a rotate-in-place,
            # and dropping the yaw here left the vehicle turning-to-face-goal forever: 209 consecutive
            # return holds with the heading unchanged, so the mission never came home (independent
            # review, "production runner drops hold yaw"). A GUARD hold stays a pure hover: the guard
            # suppresses movement, and re-aiming the vehicle is not its decision.
            if executed.yaw_rad is not None and executed.issued_by != "guard":
                rotate = getattr(self.adapter, "issue_rotate_to_yaw", None)
                if callable(rotate):
                    rotate(float(executed.yaw_rad), timeout_s=dt * 2.0)
                else:  # pragma: no cover - an adapter without rotation support
                    self.adapter.issue_hold()
            else:
                self.adapter.issue_hold()
        elif kind == "land":
            self.adapter.issue_land()
        elif kind == "takeoff":
            self.adapter.issue_takeoff()
        elif kind == "return_to_launch":
            home = Vec3(x=mission.home.x, y=mission.home.y, z=mission.home.z - mission.cruise_altitude_m)
            self.adapter.move_to(home, speed_mps=mission.cruise_speed_mps, duration_s=dt)
        elif kind == "noop":
            pass  # A completed, disarmed landing must not issue another native hover command.
        elif kind in {"arm", "abort"}:
            self.adapter.issue_hold()
        else:  # pragma: no cover - CommandKind is a closed literal
            raise ValueError(f"unsupported command kind {kind!r}")

        events.append(TruthEvent(
            sim_time_s=now_s, kind="command_executed",
            detail=f"{kind} by {executed.issued_by}",
            payload={
                "kind": kind,
                "step_index": step_index,
                "intervention_id": (
                    guard.interventions[-1]["intervention_id"]
                    if executed.issued_by == "guard" and guard.interventions else None
                ),
                "token_id": broker.view(now_s).token_id if kind == "inspect_capture" else None,
                "issued_by": executed.issued_by,
                "target": executed.target.model_dump() if executed.target else None,
                "speed_mps": executed.speed_mps,
                "reason": executed.reason,
                "overridden": executed is not command,
            },
        ))
        return executed

    def _returned_home(self, samples: list[TruthSample], manifest: ScenarioManifest) -> bool:
        if not samples:
            return False
        last = samples[-1]
        home = self.protocol.mission.home
        return last.position.horizontal_distance_to(home) <= self.protocol.mission.return_tolerance_m


def unknown_fraction(record: EpisodeRecord) -> float:
    """Share of recorded monitor steps whose verdict was UNKNOWN (diagnostic convenience)."""
    reports = [s.monitor_report for s in record.steps if s.monitor_report is not None]
    if not reports:
        return 0.0
    return sum(1 for r in reports if r.verdict is Verdict.UNKNOWN) / len(reports)
