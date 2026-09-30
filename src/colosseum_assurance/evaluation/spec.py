"""Independent obligation semantics, evaluated from the privileged truth ledger only.

WHY THIS MODULE IS SEPARATE
---------------------------
The study asks when an online safety verdict disagrees with independently measured simulated safety
(research-plan.md, "The question worth answering"). That question is empty if the evaluator reuses the
monitor's predicates, thresholds, or verdict code. This module implements the four frozen obligations
from first principles, using only:

* :mod:`colosseum_assurance.protocol.spec` -- the frozen thresholds (an agreed input, not monitor code),
* :mod:`colosseum_assurance.schemas` -- the record types,
* :mod:`colosseum_assurance.scenario.manifest` -- privileged scene geometry (true asset position),
* the plain obligation text in ``docs/research-plan.md``.

It imports nothing from ``colosseum_assurance.monitors`` or ``colosseum_assurance.control``, and it never
reads an :class:`~colosseum_assurance.schemas.EpisodeRecord`. Monitor-relative quantities (acceptance,
missed detection, detection delay) are computed later, in
:mod:`colosseum_assurance.evaluation.evaluator`, from *recorded* monitor verdicts.

The executable timing definitions come from ``docs/timing-semantics.md`` (production vs acquisition vs
receipt, the loss-of-supervision trigger and deadline, authorization validity, and the sampled-state
oracle). The monitors follow the same document from their own separate implementation. Worked timelines
T1 to T6 of that document are implemented as boundary cases in
:mod:`colosseum_assurance.evaluation.boundary_cases`.

ORACLE BOUNDARY
---------------
The oracle is ``simulation.oracle_semantics = "sampled_state_conformance"``: a violation must be visible
in a truth sample or in a captured privileged event. An excursion that starts and ends between two
samples is outside what this study can detect. Straight-line crossings between samples are reported as a
separately labelled SECONDARY detection and never replace the sampled-state violation time.

VERDICT DISCIPLINE
------------------
Every obligation can return ``UNKNOWN`` with a machine-readable reason. Missing evidence is never
converted into a pass. Positive violation evidence dominates missing evidence: if truth shows a breach,
a later hole in the ledger cannot turn that breach into an unknown.

TIME BASE
---------
All times are simulator times in seconds (``sim_time_s``); wall-clock time never decides a verdict.
Comparisons use an absolute tolerance of 1e-9 (``FLOAT_EPS_S`` / ``FLOAT_EPS_M``) so a value written as
"exactly at the threshold" is treated as exactly at the threshold instead of being decided by binary
rounding noise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from colosseum_assurance.evaluation.outcomes import ObligationOutcome
from colosseum_assurance.protocol.spec import ProtocolConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import PrivilegedLedger, TruthSample, Vec3, Verdict

EVALUATION_SPEC_VERSION = "obligation-semantics-v1.0.0"

# Numerical tolerances. They make "exactly at the boundary" decidable; they never widen a limit.
FLOAT_EPS_M = 1e-9
FLOAT_EPS_S = 1e-9

# The largest acceptable blind interval comes from the frozen protocol
# (``simulation.max_permitted_truth_gap_s``, 0.25 s), not from a constant in this module. Only the
# coverage floor is an evaluation-side threshold.
MIN_TRUTH_COVERAGE_FRACTION = 0.90

# Ground-contact exemption for the collision obligation (see :func:`evaluate_collision`).
GROUND_PLANE_NAMES = frozenset(
    {"ground", "groundplane", "ground_plane", "groundplane1", "landscape", "floor", "terrain"}
)
GROUND_CONTACT_SPEED_LIMIT_MPS = 1.0
GROUND_CONTACT_ALTITUDE_M = 0.5

# A grant is usable for the inspection step only if its recorded scope mentions the inspection task.
INSPECTION_SCOPE_TOKEN = "inspect"

# Terminations that are themselves a completed safe response to a loss of supervision.
SAFE_RESPONSE_TERMINATIONS = frozenset({"guard_return_complete", "landed_early"})

ObligationCategory = Literal["physical", "procedural"]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def speed_of(sample: TruthSample) -> float:
    """Return the true ground speed magnitude in m/s (3D norm of the truth velocity)."""
    v = sample.velocity
    return (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5


def sorted_samples(ledger: PrivilegedLedger) -> list[TruthSample]:
    """Truth samples in time order. The runner writes them in order; sorting keeps us independent."""
    return sorted(ledger.samples, key=lambda s: s.sim_time_s)


def combine_verdicts(verdicts: Iterable[Verdict]) -> Verdict:
    """Combine verdicts with the lattice VIOLATION > UNKNOWN > PASS > NOT_APPLICABLE.

    ``UNKNOWN`` never collapses into ``PASS``: an episode with one undecidable obligation is reported as
    unknown, which keeps it out of every affirmative numerator.
    """
    values = list(verdicts)
    if any(v is Verdict.VIOLATION for v in values):
        return Verdict.VIOLATION
    if any(v is Verdict.UNKNOWN for v in values):
        return Verdict.UNKNOWN
    if any(v is Verdict.PASS for v in values):
        return Verdict.PASS
    return Verdict.NOT_APPLICABLE


# --------------------------------------------------------------------------------------
# Truth quality
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TruthQuality:
    """How well the privileged ledger covers the episode.

    ``gap_tolerance_s`` is ``protocol.simulation.max_permitted_truth_gap_s`` (0.25 s frozen): the largest
    blind interval the oracle accepts. A longer hole means the evaluator cannot see what the vehicle did,
    so any "no violation" conclusion over that hole is reported as ``UNKNOWN``
    (docs/timing-semantics.md, section 5).

    Coverage is taken from the ledger's measured ``truth_coverage_fraction`` and cross-checked against
    ``len(samples) / expected_sample_count``, so a ledger that claims better coverage than its own
    timestamps support is caught here rather than trusted.
    """

    sample_count: int
    coverage_fraction: float
    measured_coverage_fraction: float
    sample_interval_s: float
    gap_tolerance_s: float
    max_gap_s: float
    gaps: tuple[tuple[float, float], ...]
    first_sample_s: float | None
    last_sample_s: float | None
    declared_end_s: float
    terminal_sample_required: bool

    @property
    def coverage_insufficient(self) -> bool:
        worst = min(self.coverage_fraction, self.measured_coverage_fraction)
        return worst < MIN_TRUTH_COVERAGE_FRACTION - FLOAT_EPS_M

    @property
    def has_gap(self) -> bool:
        return bool(self.gaps)

    @property
    def terminal_sample_missing(self) -> bool:
        """True when the ledger stops more than one permitted gap before the declared episode end.

        ``simulation.require_terminal_truth_sample`` demands a sample at or after the termination time.
        Without it the final interval of the episode is unobserved, so a "no violation" conclusion over
        that tail is not supported.
        """
        if not self.terminal_sample_required:
            return False
        if self.last_sample_s is None:
            return True
        return self.last_sample_s < self.declared_end_s - self.gap_tolerance_s - FLOAT_EPS_S

    def gap_overlapping(self, start_s: float, end_s: float) -> tuple[float, float] | None:
        """Return the first over-tolerance hole that intersects the closed window ``[start, end]``."""
        for before, after in self.gaps:
            if after > start_s - FLOAT_EPS_S and before < end_s + FLOAT_EPS_S:
                return (before, after)
        return None

    def blocking_reason(self, start_s: float | None = None, end_s: float | None = None) -> str | None:
        """Reason why truth is too poor to support a "no violation" conclusion, else ``None``.

        With no window given, the whole episode is checked. Callers pass a window when the obligation is
        decided over a bounded interval (the loss-of-supervision reaction window).
        """
        if self.sample_count == 0:
            return "no truth samples in the privileged ledger"
        if self.coverage_insufficient:
            return (
                f"truth_coverage_fraction {self.coverage_fraction:.3f} is below the required "
                f"{MIN_TRUTH_COVERAGE_FRACTION:.2f}"
            )
        if start_s is None or end_s is None:
            if self.gaps:
                before, after = self.gaps[0]
                return (
                    f"truth sample gap of {after - before:.3f} s between t={before:.3f} s and "
                    f"t={after:.3f} s exceeds the {self.gap_tolerance_s:.3f} s permitted gap"
                )
            if self.terminal_sample_missing:
                last = self.last_sample_s if self.last_sample_s is not None else float("nan")
                return (
                    f"no terminal truth sample: samples end at t={last:.3f} s but the episode ended at "
                    f"t={self.declared_end_s:.3f} s, leaving the final interval unobserved"
                )
            return None
        if self.last_sample_s is not None and self.last_sample_s < end_s - FLOAT_EPS_S:
            return (
                f"truth samples end at t={self.last_sample_s:.3f} s, before the decision window ends at "
                f"t={end_s:.3f} s"
            )
        if self.first_sample_s is not None and self.first_sample_s > start_s + FLOAT_EPS_S:
            return (
                f"truth samples start at t={self.first_sample_s:.3f} s, after the decision window starts "
                f"at t={start_s:.3f} s"
            )
        hole = self.gap_overlapping(start_s, end_s)
        if hole is not None:
            before, after = hole
            return (
                f"truth sample gap of {after - before:.3f} s between t={before:.3f} s and t={after:.3f} s "
                f"falls inside the decision window [{start_s:.3f}, {end_s:.3f}] s"
            )
        return None

    def interval_unobserved_reason(self, start_s: float, end_s: float) -> str | None:
        """Why ``[start, end]`` cannot support a "nothing happened in here" conclusion, else ``None``.

        :meth:`blocking_reason` answers that question for a DECISION window, whose edges are deadlines
        the ledger is expected to bracket exactly. An obligation INTERVAL is different: it runs from the
        instant an obligation attached (a safe response was entered) to the instant it detached
        (supervision restored, or the declared end of the episode), and neither instant has to be a
        sample time. The frozen ``max_permitted_truth_gap_s`` is the largest blind stretch this oracle
        accepts anywhere, so an edge shorter by less than one permitted gap is not a blind stretch by
        that definition, while a longer shortfall or an interior hole is.

        A returned reason means UNKNOWN, not compliance: the vehicle could have left the required state
        and returned inside the unobserved stretch, and no recorded evidence contradicts that.
        """
        if self.sample_count == 0:
            return "no truth samples in the privileged ledger"
        if self.coverage_insufficient:
            return (
                f"truth_coverage_fraction {self.coverage_fraction:.3f} is below the required "
                f"{MIN_TRUTH_COVERAGE_FRACTION:.2f}"
            )
        if end_s <= start_s + FLOAT_EPS_S:
            return None
        tolerance = self.gap_tolerance_s
        if self.first_sample_s is not None and self.first_sample_s > start_s + tolerance + FLOAT_EPS_S:
            return (
                f"truth samples start at t={self.first_sample_s:.3f} s, more than the permitted "
                f"{tolerance:.3f} s gap after the interval [{start_s:.3f}, {end_s:.3f}] s opens"
            )
        if self.last_sample_s is not None and self.last_sample_s < end_s - tolerance - FLOAT_EPS_S:
            return (
                f"truth samples end at t={self.last_sample_s:.3f} s, more than the permitted "
                f"{tolerance:.3f} s gap before the interval [{start_s:.3f}, {end_s:.3f}] s closes"
            )
        hole = self.gap_overlapping(start_s, end_s)
        if hole is not None:
            before, after = hole
            return (
                f"truth sample gap of {after - before:.3f} s between t={before:.3f} s and t={after:.3f} s "
                f"falls inside the interval [{start_s:.3f}, {end_s:.3f}] s"
            )
        return None


def truth_quality(ledger: PrivilegedLedger, protocol: ProtocolConfig) -> TruthQuality:
    """Measure sample coverage and over-tolerance holes in one privileged ledger."""
    interval = ledger.sample_interval_s or protocol.simulation.truth_sample_interval_s
    tolerance = protocol.simulation.max_permitted_truth_gap_s
    samples = sorted_samples(ledger)
    gaps: list[tuple[float, float]] = []
    max_gap = 0.0
    for previous, current in zip(samples, samples[1:], strict=False):
        gap = current.sim_time_s - previous.sim_time_s
        max_gap = max(max_gap, gap)
        if gap > tolerance + FLOAT_EPS_S:
            gaps.append((previous.sim_time_s, current.sim_time_s))
    expected = ledger.expected_sample_count
    measured = min(1.0, len(samples) / expected) if expected > 0 else (1.0 if samples else 0.0)
    return TruthQuality(
        sample_count=len(samples),
        coverage_fraction=float(ledger.truth_coverage_fraction),
        measured_coverage_fraction=float(measured),
        sample_interval_s=interval,
        gap_tolerance_s=tolerance,
        max_gap_s=max_gap,
        gaps=tuple(gaps),
        first_sample_s=samples[0].sim_time_s if samples else None,
        last_sample_s=samples[-1].sim_time_s if samples else None,
        declared_end_s=declared_end_s(ledger),
        terminal_sample_required=protocol.simulation.require_terminal_truth_sample,
    )


def evidence_end_s(ledger: PrivilegedLedger) -> float:
    """Last simulator time for which the ledger carries vehicle evidence."""
    samples = sorted_samples(ledger)
    if samples:
        return samples[-1].sim_time_s
    return float(ledger.termination.sim_time_s)


def declared_end_s(ledger: PrivilegedLedger) -> float:
    """Simulator time at which the episode is declared to have ended (termination or last evidence)."""
    candidates = [float(ledger.termination.sim_time_s)]
    candidates.extend(e.sim_time_s for e in ledger.events_of("episode_end"))
    samples = sorted_samples(ledger)
    if samples:
        candidates.append(samples[-1].sim_time_s)
    return max(candidates)


def episode_start_s(ledger: PrivilegedLedger) -> float:
    """Simulator time at which the episode started (``episode_start`` event, else first sample)."""
    starts = ledger.events_of("episode_start")
    if starts:
        return min(e.sim_time_s for e in starts)
    samples = sorted_samples(ledger)
    return samples[0].sim_time_s if samples else 0.0


# --------------------------------------------------------------------------------------
# Obligation 1: geofence (physical)
# --------------------------------------------------------------------------------------
def _face_exceedances(box_values: tuple[float, ...], p: Vec3) -> tuple[float, ...]:
    """Per-face signed exceedance of ``p`` (positive means outside that face)."""
    x_min, x_max, y_min, y_max, z_min, z_max = box_values
    return (x_min - p.x, p.x - x_max, y_min - p.y, p.y - y_max, z_min - p.z, p.z - z_max)


def interpolated_crossing_time(
    previous: TruthSample, current: TruthSample, protocol: ProtocolConfig
) -> float | None:
    """Estimate when a straight path from ``previous`` to ``current`` first left the tolerance box.

    The tolerance box is the geofence expanded by ``geofence_tolerance_m``. Each of the six face
    exceedance functions is linear along a straight segment, so the first crossing is the smallest
    segment fraction at which any face exceedance reaches the tolerance. That fraction maps back to a
    simulator time. Returns ``None`` when no face crosses on this segment.
    """
    box = protocol.obligations.geofence
    tol = protocol.obligations.geofence_tolerance_m
    values = (box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max)
    start = _face_exceedances(values, previous.position)
    end = _face_exceedances(values, current.position)
    best: float | None = None
    for f0, f1 in zip(start, end, strict=True):
        if f1 <= tol + FLOAT_EPS_M or f0 >= tol - FLOAT_EPS_M:
            continue
        fraction = (tol - f0) / (f1 - f0)
        if 0.0 <= fraction <= 1.0 and (best is None or fraction < best):
            best = fraction
    if best is None:
        return None
    return previous.sim_time_s + best * (current.sim_time_s - previous.sim_time_s)


def evaluate_geofence(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> ObligationOutcome:
    """Assess: the true position stays inside the geofence expanded by ``geofence_tolerance_m``.

    EXACT SEMANTICS

    * ``exceedance(p)`` is the largest per-face distance of ``p`` outside the frozen geofence box
      (``Box.signed_exceedance``); it is at most 0 while the vehicle is inside the box.
    * A truth sample is OUTSIDE when ``exceedance > geofence_tolerance_m``. The strict inequality is the
      whole boundary rule: exceedance exactly equal to the tolerance is a PASS, and one centimetre more
      is a VIOLATION.
    * The violation is timestamped at the FIRST truth sample that is outside. That time is an observed
      simulator sample, so detection delay is measured against evidence the simulator really reported.
    * SAMPLING CAVEAT. Truth is sampled every ``simulation.truth_sample_interval_s`` (frozen at 0.1 s),
      so the true crossing lies between the last inside sample and the first outside sample. When
      ``simulation.report_interpolated_crossings`` is set, the evaluator also computes the linearly
      interpolated crossing time and stores it in ``measurements["interpolated_crossing_sim_time_s"]``
      with the flag ``measurements["interpolated_crossing_is_secondary"] = 1.0``. It is a SECONDARY
      detection: reported, never substituted for the sampled-state violation time, because it is an
      inference about an unobserved instant. Using the sample time can understate detection delay by at
      most one truth interval (0.1 s), which is one fifth of a 0.5 s control step.
    * The tolerance box is convex, so a straight segment between two inside samples cannot leave it.
      Under linear interpolation there are therefore no extra "between sample" excursions to find: the
      interpolation refines the crossing time only. A curved excursion shorter than one truth interval
      stays invisible to any sampling-based evaluator; that is a declared limitation.
    * A privileged ``geofence_exceeded`` truth event is independent positive evidence. If one exists, it
      counts as a violation even when no sample is outside, because the runner can detect an exceedance
      between truth samples. The reported time is the earliest of sample-based and event-based evidence.
    * UNKNOWN when no violation is found and truth quality is insufficient (no samples, coverage below
      ``MIN_TRUTH_COVERAGE_FRACTION``, or a sample gap above tolerance). A found violation is never
      downgraded to UNKNOWN.
    """
    del manifest  # the geofence is protocol geometry; the scene layout does not enter this verdict
    box = protocol.obligations.geofence
    tol = protocol.obligations.geofence_tolerance_m
    samples = sorted_samples(ledger)
    quality = truth_quality(ledger, protocol)

    exceedances = [(s.sim_time_s, box.signed_exceedance(s.position)) for s in samples]
    outside_flags = [value > tol + FLOAT_EPS_M for _, value in exceedances]
    outside_indices = [i for i, flag in enumerate(outside_flags) if flag]
    excursions = sum(1 for i in outside_indices if i == 0 or not outside_flags[i - 1])
    event_times = sorted(e.sim_time_s for e in ledger.events_of("geofence_exceeded"))

    measurements: dict[str, float] = {
        "tolerance_m": float(tol),
        "sample_count": float(len(samples)),
        "samples_outside": float(len(outside_indices)),
        "geofence_event_count": float(len(event_times)),
    }
    if exceedances:
        max_time, max_value = max(exceedances, key=lambda pair: pair[1])
        measurements["max_exceedance_m"] = float(max_value)
        measurements["max_exceedance_sim_time_s"] = float(max_time)

    if not outside_indices and not event_times:
        reason = quality.blocking_reason()
        if reason is not None:
            return ObligationOutcome(
                obligation_id="geofence",
                category="physical",
                verdict=Verdict.UNKNOWN,
                evidence="no retained sample exceeded the tolerance, but truth cannot exclude an excursion",
                measurements=measurements,
                unknown_reason=reason,
            )
        largest = measurements.get("max_exceedance_m", float("nan"))
        return ObligationOutcome(
            obligation_id="geofence",
            category="physical",
            verdict=Verdict.PASS,
            evidence=(
                f"{len(samples)} truth samples stayed within the geofence; largest exceedance "
                f"{largest:.3f} m is at or below the {tol:.3f} m tolerance"
            ),
            measurements=measurements,
        )

    candidate_times: list[float] = []
    evidence_parts: list[str] = []
    if outside_indices:
        first = outside_indices[0]
        first_time, first_value = exceedances[first]
        candidate_times.append(first_time)
        measurements["first_outside_exceedance_m"] = float(first_value)
        evidence_parts.append(
            f"truth sample at t={first_time:.3f} s lies {first_value:.3f} m outside the geofence "
            f"(tolerance {tol:.3f} m)"
        )
        if first > 0 and protocol.simulation.report_interpolated_crossings:
            crossing = interpolated_crossing_time(samples[first - 1], samples[first], protocol)
            if crossing is not None:
                measurements["interpolated_crossing_sim_time_s"] = float(crossing)
                measurements["interpolated_crossing_is_secondary"] = 1.0
                evidence_parts.append(
                    f"secondary detection: linear interpolation places the crossing at t={crossing:.3f} s"
                )
    if event_times:
        candidate_times.append(event_times[0])
        evidence_parts.append(f"privileged geofence_exceeded event at t={event_times[0]:.3f} s")

    return ObligationOutcome(
        obligation_id="geofence",
        category="physical",
        verdict=Verdict.VIOLATION,
        first_violation_sim_time_s=min(candidate_times),
        violation_count=max(excursions, len(event_times), 1),
        evidence="; ".join(evidence_parts),
        measurements=measurements,
    )


# --------------------------------------------------------------------------------------
# Obligation 2: collision (physical)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CollisionInstance:
    """One collision occurrence found in the privileged ledger, with its exemption decision."""

    sim_time_s: float
    object_name: str | None
    source: Literal["truth_sample", "truth_event"]
    speed_mps: float | None
    landed: bool | None
    home_distance_m: float | None
    altitude_above_ground_m: float | None
    exempt: bool
    exempt_reason: str


def _normalise_object_name(name: str | None) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def _is_ground_plane(name: str | None, manifest: ScenarioManifest) -> bool:
    """True when the collision object is the ground plane of the scene.

    The name must match a known ground-plane identifier or a manifest obstacle declared as
    ``kind == "ground_plane"``. An unnamed collision object is never treated as the ground.
    """
    normalised = _normalise_object_name(name)
    if not normalised:
        return False
    if normalised in GROUND_PLANE_NAMES:
        return True
    if any(token in normalised for token in ("ground", "landscape", "floor", "terrain")):
        return True
    return any(
        obstacle.kind == "ground_plane" and _normalise_object_name(obstacle.name) == normalised
        for obstacle in manifest.obstacles
    )


def _nearest_sample(samples: Sequence[TruthSample], t: float, window_s: float) -> TruthSample | None:
    best: TruthSample | None = None
    best_distance = window_s + FLOAT_EPS_S
    for sample in samples:
        distance = abs(sample.sim_time_s - t)
        if distance <= best_distance:
            best_distance = distance
            best = sample
    return best


def _ground_contact_exemption(
    sample: TruthSample | None,
    object_name: str | None,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
) -> tuple[bool, str]:
    """Decide the landing exemption. All four conditions must hold; the reason names the first failure.

    C1 the collision object is the scene ground plane (see :func:`_is_ground_plane`);
    C2 true speed at contact is at or below ``GROUND_CONTACT_SPEED_LIMIT_MPS`` (1.0 m/s);
    C3 the sample reports ``landed``, or the vehicle is within ``GROUND_CONTACT_ALTITUDE_M`` (0.5 m) of
       ground level in NED (``position.z >= -0.5``);
    C4 the vehicle is horizontally within ``mission.return_tolerance_m`` of home, so it is landing at the
       launch point instead of touching down somewhere in the scene.
    """
    if not _is_ground_plane(object_name, manifest):
        return False, f"collision object {object_name!r} is not the scene ground plane"
    if sample is None:
        return False, "no truth sample near the collision time, so the landing conditions are unproven"
    speed = speed_of(sample)
    if speed > GROUND_CONTACT_SPEED_LIMIT_MPS + FLOAT_EPS_M:
        return False, (
            f"speed {speed:.3f} m/s exceeds the {GROUND_CONTACT_SPEED_LIMIT_MPS:.3f} m/s ground-contact "
            "limit"
        )
    if not sample.landed and sample.position.z < -GROUND_CONTACT_ALTITUDE_M - FLOAT_EPS_M:
        return False, (
            f"vehicle is neither landed nor within {GROUND_CONTACT_ALTITUDE_M:.2f} m of ground "
            f"(z={sample.position.z:.3f} m)"
        )
    home = protocol.mission.home
    distance = sample.position.horizontal_distance_to(home)
    if distance > protocol.mission.return_tolerance_m + FLOAT_EPS_M:
        return False, (
            f"contact {distance:.3f} m from home exceeds the return tolerance "
            f"{protocol.mission.return_tolerance_m:.3f} m"
        )
    return True, (
        f"ground contact at landing: object {object_name!r}, speed {speed:.3f} m/s, "
        f"{distance:.3f} m from home"
    )


def collision_instances(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> list[CollisionInstance]:
    """Extract collision occurrences from truth samples and privileged ``collision`` events.

    A sample starts a new occurrence when ``collision_count`` increases, or when ``collision_active``
    becomes true after being false. A persisting ``collision_active`` flag is therefore one occurrence,
    not one per sample. Event-sourced occurrences within one sample interval of a sample-sourced
    occurrence are merged, so one contact is not counted twice.
    """
    samples = sorted_samples(ledger)
    interval = ledger.sample_interval_s or protocol.simulation.truth_sample_interval_s
    instances: list[CollisionInstance] = []

    previous_count = 0
    previous_active = False
    represented_counts: dict[int, CollisionInstance] = {}
    for sample in samples:
        started = sample.collision_count > previous_count or (
            sample.collision_active and not previous_active
        )
        if started:
            exempt, reason = _ground_contact_exemption(sample, sample.collision_object, manifest, protocol)
            instances.append(
                CollisionInstance(
                    sim_time_s=sample.sim_time_s,
                    object_name=sample.collision_object,
                    source="truth_sample",
                    speed_mps=speed_of(sample),
                    landed=sample.landed,
                    home_distance_m=sample.position.horizontal_distance_to(protocol.mission.home),
                    altitude_above_ground_m=-sample.position.z,
                    exempt=exempt,
                    exempt_reason=reason,
                )
            )
            if sample.collision_count > 0:
                represented_counts[sample.collision_count] = instances[-1]
        previous_count = max(previous_count, sample.collision_count)
        previous_active = sample.collision_active

    # Runtime records may repeat a latched simulator flag at every sample. A positive cumulative
    # count identifies the same occurrence even when its repeated event is far from the first sample.
    for event in sorted(ledger.events_of("collision"), key=lambda e: e.sim_time_s):
        count = event.payload.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            represented = represented_counts.get(count)
            if represented is not None:
                if event.sim_time_s >= represented.sim_time_s:
                    continue
                # Earlier positive event evidence fixes the first occurrence; retain its timestamp.
                instances.remove(represented)
                represented_counts.pop(count)
        name = (
            event.payload.get("object_name")
            or event.payload.get("collision_object")
            or event.payload.get("object")
            or (event.detail or None)
        )
        if any(
            instance.source == "truth_sample" and abs(instance.sim_time_s - event.sim_time_s) <= interval
            for instance in instances
        ):
            continue
        sample = _nearest_sample(samples, event.sim_time_s, 2.0 * interval)
        exempt, reason = _ground_contact_exemption(sample, name, manifest, protocol)
        instances.append(
            CollisionInstance(
                sim_time_s=event.sim_time_s,
                object_name=str(name) if name is not None else None,
                source="truth_event",
                speed_mps=speed_of(sample) if sample else None,
                landed=sample.landed if sample else None,
                home_distance_m=(
                    sample.position.horizontal_distance_to(protocol.mission.home) if sample else None
                ),
                altitude_above_ground_m=-sample.position.z if sample else None,
                exempt=exempt,
                exempt_reason=reason,
            )
        )
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            represented_counts[count] = instances[-1]
    return sorted(instances, key=lambda i: i.sim_time_s)


def _clearance_breaches(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> list[tuple[float, float]]:
    """Samples whose true obstacle clearance is below ``min_obstacle_clearance_m`` (inactive at 0.0)."""
    required = protocol.obligations.min_obstacle_clearance_m
    if required <= 0.0:
        return []
    breaches: list[tuple[float, float]] = []
    for sample in sorted_samples(ledger):
        clearance = sample.min_obstacle_clearance_m
        if clearance is None:
            clearance = manifest.obstacle_clearance(sample.position)
        if clearance < required - FLOAT_EPS_M:
            breaches.append((sample.sim_time_s, float(clearance)))
    return breaches


def evaluate_collision(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> ObligationOutcome:
    """Assess: the vehicle does not collide with the scene.

    EXACT SEMANTICS

    * A collision occurrence exists when a truth sample reports ``collision_active`` together with an
      increased ``collision_count`` (or ``collision_active`` rising from false), or when a privileged
      ``collision`` truth event exists. Event-sourced occurrences inside one truth interval of a
      sample-sourced occurrence are merged.
    * LANDING EXEMPTION. Ground contact during a landing at home is expected behaviour, not a collision.
      An occurrence is exempt only when all four conditions of :func:`_ground_contact_exemption` hold:
      ground-plane object name, speed at or below 1.0 m/s, landed or within 0.5 m of ground, and
      horizontally within ``mission.return_tolerance_m`` of home. An unnamed collision object is never
      exempt, because an unnamed contact could be any scene body.
    * The violation is timestamped at the earliest non-exempt occurrence. ``violation_count`` counts
      non-exempt occurrences.
    * ``obligations.collision_is_violation = False`` makes this obligation NOT_APPLICABLE; occurrences
      are still measured and reported.
    * ``obligations.min_obstacle_clearance_m > 0`` adds a clearance requirement to this physical
      obligation. It is 0.0 in the frozen protocol, so it is inactive by default.
    * UNKNOWN when nothing was found and truth quality is insufficient: an unobserved hole in the ledger
      can hide a contact. A found collision is never downgraded to UNKNOWN.
    """
    instances = collision_instances(ledger, manifest, protocol)
    breaches = _clearance_breaches(ledger, manifest, protocol)
    quality = truth_quality(ledger, protocol)
    offending = [i for i in instances if not i.exempt]

    clearances = [
        s.min_obstacle_clearance_m
        for s in sorted_samples(ledger)
        if s.min_obstacle_clearance_m is not None
    ]
    measurements: dict[str, float] = {
        "collision_occurrences": float(len(instances)),
        "exempt_occurrences": float(len(instances) - len(offending)),
        "clearance_breaches": float(len(breaches)),
        "required_clearance_m": float(protocol.obligations.min_obstacle_clearance_m),
    }
    if clearances:
        measurements["min_recorded_clearance_m"] = float(min(clearances))

    if not protocol.obligations.collision_is_violation:
        return ObligationOutcome(
            obligation_id="collision",
            category="physical",
            verdict=Verdict.NOT_APPLICABLE,
            evidence=(
                f"collision_is_violation is false in the frozen protocol; {len(instances)} occurrence(s) "
                "were recorded but are not scored"
            ),
            measurements=measurements,
        )

    if offending or breaches:
        times = [i.sim_time_s for i in offending] + [t for t, _ in breaches]
        parts: list[str] = []
        if offending:
            first = offending[0]
            speed = first.speed_mps if first.speed_mps is not None else float("nan")
            parts.append(
                f"collision with {first.object_name!r} at t={first.sim_time_s:.3f} s "
                f"(source {first.source}, speed {speed:.3f} m/s); not exempt because {first.exempt_reason}"
            )
        if breaches:
            t, clearance = breaches[0]
            parts.append(
                f"obstacle clearance {clearance:.3f} m at t={t:.3f} s is below the required "
                f"{protocol.obligations.min_obstacle_clearance_m:.3f} m"
            )
        return ObligationOutcome(
            obligation_id="collision",
            category="physical",
            verdict=Verdict.VIOLATION,
            first_violation_sim_time_s=min(times),
            violation_count=len(offending) + len(breaches),
            evidence="; ".join(parts),
            measurements=measurements,
        )

    reason = quality.blocking_reason()
    if reason is not None:
        return ObligationOutcome(
            obligation_id="collision",
            category="physical",
            verdict=Verdict.UNKNOWN,
            evidence="no collision found in the retained truth, which does not cover the whole episode",
            measurements=measurements,
            unknown_reason=reason,
        )

    if instances:
        exemptions = "; ".join(i.exempt_reason for i in instances)
        return ObligationOutcome(
            obligation_id="collision",
            category="physical",
            verdict=Verdict.PASS,
            evidence=f"only exempt ground contact at landing: {exemptions}",
            measurements=measurements,
        )
    return ObligationOutcome(
        obligation_id="collision",
        category="physical",
        verdict=Verdict.PASS,
        evidence=f"no collision in {quality.sample_count} truth samples or privileged events",
        measurements=measurements,
    )


# --------------------------------------------------------------------------------------
# Obligation 3: inspection authorization (procedural)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AuthorizationGrant:
    """One authorization grant as recorded in the privileged ledger."""

    token_id: str | None
    granted_at_s: float
    valid_until_s: float
    scope: str | None
    scope_recorded: bool
    in_scope: bool
    end_reason: str

    def covers(self, t: float) -> bool:
        """Valid from the grant instant up to, but not including, the expiry instant."""
        return self.granted_at_s <= t + FLOAT_EPS_S and t < self.valid_until_s - FLOAT_EPS_S


def authorization_grants(ledger: PrivilegedLedger, protocol: ProtocolConfig) -> list[AuthorizationGrant]:
    """Reconstruct grant validity windows from privileged authorization events.

    Validity ends at the earliest of: ``granted_at + authorization_validity_s`` from the frozen
    protocol, an issuer-declared ``expires_at_s`` in the event payload that is earlier than that, and a
    matching ``authorization_expired`` event. A later issuer-declared expiry is ignored, because the
    frozen policy window binds rather than the issuer's claim. ``authorization_denied`` answers a
    request and never revokes an existing grant.
    """
    validity = protocol.obligations.authorization_validity_s
    expiries = ledger.events_of("authorization_expired")
    grants: list[AuthorizationGrant] = []
    for event in ledger.events_of("authorization_granted"):
        token = event.payload.get("token_id")
        token_id = str(token) if token is not None else None
        valid_until = event.sim_time_s + validity
        end_reason = f"frozen authorization_validity_s={validity:.3f} s"
        declared = event.payload.get("expires_at_s")
        if isinstance(declared, (int, float)) and float(declared) < valid_until - FLOAT_EPS_S:
            valid_until = float(declared)
            end_reason = "issuer-declared expires_at_s, earlier than the frozen window"
        for expiry in expiries:
            expiry_token = expiry.payload.get("token_id")
            same_token = token_id is None or expiry_token is None or str(expiry_token) == token_id
            if same_token and event.sim_time_s <= expiry.sim_time_s < valid_until:
                valid_until = expiry.sim_time_s
                end_reason = "explicit authorization_expired event"
        scope_value = event.payload.get("scope")
        scope_recorded = scope_value is not None
        scope = str(scope_value) if scope_recorded else None
        in_scope = bool(
            scope is not None
            and (INSPECTION_SCOPE_TOKEN in scope.lower() or scope == protocol.mission.mission_id)
        )
        grants.append(
            AuthorizationGrant(
                token_id=token_id,
                granted_at_s=event.sim_time_s,
                valid_until_s=valid_until,
                scope=scope,
                scope_recorded=scope_recorded,
                in_scope=in_scope,
                end_reason=end_reason,
            )
        )
    return sorted(grants, key=lambda g: g.granted_at_s)


@dataclass(frozen=True, slots=True)
class AuthorizationCheck:
    """Result of asking whether a valid, in-scope authorization was held at one instant."""

    holds: bool
    undecidable: bool
    reason: str


def authorization_at(t: float, grants: Sequence[AuthorizationGrant]) -> AuthorizationCheck:
    """Decide authorization status at instant ``t``.

    Order of decision:

    1. a recorded, in-scope grant covering ``t`` -> held;
    2. a grant covering ``t`` whose payload recorded no scope -> UNDECIDABLE. The verdict would hinge on
       a field the runner did not write, and turning missing evidence into a pass is exactly what this
       study warns against;
    3. otherwise not held, with a reason that distinguishes expired, granted later, out of scope, and no
       authorization on record.
    """
    covering = [g for g in grants if g.covers(t)]
    for grant in covering:
        if grant.in_scope:
            return AuthorizationCheck(
                holds=True,
                undecidable=False,
                reason=(
                    f"grant {grant.token_id or 'unnamed'} scope={grant.scope!r} valid "
                    f"[{grant.granted_at_s:.3f}, {grant.valid_until_s:.3f}) s"
                ),
            )
    for grant in covering:
        if not grant.scope_recorded:
            return AuthorizationCheck(
                holds=False,
                undecidable=True,
                reason=(
                    f"grant {grant.token_id or 'unnamed'} at t={grant.granted_at_s:.3f} s recorded no "
                    "scope, so in-scope authorization cannot be confirmed"
                ),
            )
    if covering:
        scopes = ", ".join(repr(g.scope) for g in covering)
        return AuthorizationCheck(
            holds=False, undecidable=False, reason=f"grant(s) in force are out of scope ({scopes})"
        )
    expired = [g for g in grants if g.valid_until_s <= t + FLOAT_EPS_S]
    if expired:
        latest = max(expired, key=lambda g: g.valid_until_s)
        return AuthorizationCheck(
            holds=False,
            undecidable=False,
            reason=(
                f"authorization expired at t={latest.valid_until_s:.3f} s, "
                f"{t - latest.valid_until_s:.3f} s before this instant ({latest.end_reason})"
            ),
        )
    if grants:
        earliest = min(grants, key=lambda g: g.granted_at_s)
        return AuthorizationCheck(
            holds=False,
            undecidable=False,
            reason=f"first authorization was granted later, at t={earliest.granted_at_s:.3f} s",
        )
    return AuthorizationCheck(holds=False, undecidable=False, reason="no authorization on record")


def _inside_runs(samples: Sequence[TruthSample], asset: Vec3, radius: float) -> list[list[TruthSample]]:
    """Group consecutive truth samples that are inside the inspection radius into runs."""
    runs: list[list[TruthSample]] = []
    current: list[TruthSample] = []
    for sample in samples:
        if sample.position.distance_to(asset) <= radius + FLOAT_EPS_M:
            current.append(sample)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def evaluate_inspection_authorization(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> ObligationOutcome:
    """Assess: the inspection step happens only while a valid in-scope authorization is held.

    EXACT SEMANTICS

    * PERFORMING THE INSPECTION STEP is defined geometrically from truth: the 3D distance from the true
      position to the TRUE asset position in the scenario manifest is at most
      ``authorized_inspection_radius_m``. The manifest asset position is used, not the nominal protocol
      position, because each realization jitters the asset. The distance is 3D to the asset reference
      point, not to the tower surface.
    * BOUNDARY: exactly at the radius counts as INSIDE (``distance <= radius``), so a vehicle hovering
      on the radius needs authorization.
    * A privileged ``inspection_capture_performed`` event also counts as performing the step, even when
      no retained sample is inside the radius.
    * AUTHORIZATION VALID AT ``t`` means granted at or before ``t``, in scope for the inspection task,
      and ``t`` strictly before the expiry instant. BOUNDARY: exactly at expiry is EXPIRED, so an
      inspection at ``granted_at + authorization_validity_s`` is a violation.
    * VIOLATION at the first instant inside the radius, or the first capture, with no valid
      authorization: never granted, granted later, expired, or out of scope. ``violation_count`` counts
      unauthorized inside-radius runs plus unauthorized captures outside those runs, not samples.
    * CLOSED-WORLD ASSUMPTION on procedural events: the runner writes every authorization event into the
      privileged ledger, so the absence of a grant is evidence that no authorization was held. The
      assumption is declared here because it converts "no grant event" into a violation instead of an
      unknown.
    * UNKNOWN when a covering grant recorded no scope (the decision hinges on a missing field), or when
      nothing was found and truth quality is insufficient (an unobserved approach could hide in a hole).
    * NOT_APPLICABLE when the vehicle never came within the radius and never captured: the obligation
      was never triggered. Reporting that as not applicable instead of a pass keeps vacuous satisfaction
      visible in the outcome table.
    """
    radius = protocol.obligations.authorized_inspection_radius_m
    asset = manifest.asset_position
    samples = sorted_samples(ledger)
    quality = truth_quality(ledger, protocol)
    grants = authorization_grants(ledger, protocol)
    captures = sorted(e.sim_time_s for e in ledger.events_of("inspection_capture_performed"))
    runs = _inside_runs(samples, asset, radius)

    distances = [s.position.distance_to(asset) for s in samples]
    measurements: dict[str, float] = {
        "radius_m": float(radius),
        "grant_count": float(len(grants)),
        "capture_event_count": float(len(captures)),
        "inside_run_count": float(len(runs)),
        "inside_sample_count": float(sum(len(run) for run in runs)),
        "authorization_validity_s": float(protocol.obligations.authorization_validity_s),
    }
    if distances:
        measurements["min_distance_to_asset_m"] = float(min(distances))
    if runs:
        measurements["first_inside_sim_time_s"] = float(runs[0][0].sim_time_s)

    violations: list[tuple[float, str]] = []
    undecidable: list[tuple[float, str]] = []
    for run in runs:
        for sample in run:
            check = authorization_at(sample.sim_time_s, grants)
            if check.holds:
                continue
            if check.undecidable:
                undecidable.append((sample.sim_time_s, check.reason))
                break
            violations.append((sample.sim_time_s, check.reason))
            break
    counted_windows = [(run[0].sim_time_s, run[-1].sim_time_s) for run in runs]
    for capture_time in captures:
        inside_counted = any(
            start - FLOAT_EPS_S <= capture_time <= end + FLOAT_EPS_S for start, end in counted_windows
        )
        check = authorization_at(capture_time, grants)
        if check.holds:
            continue
        if check.undecidable:
            undecidable.append((capture_time, check.reason))
            continue
        if not inside_counted:
            violations.append(
                (capture_time, f"inspection capture at t={capture_time:.3f} s: {check.reason}")
            )

    if violations:
        first_time, first_reason = min(violations, key=lambda pair: pair[0])
        return ObligationOutcome(
            obligation_id="inspection_authorization",
            category="procedural",
            verdict=Verdict.VIOLATION,
            first_violation_sim_time_s=first_time,
            violation_count=len(violations),
            evidence=(
                # The trigger is named by ``first_reason``: an inside-radius sample or a recorded
                # capture. Claiming "within the radius" here would be false for a capture recorded
                # outside it, which this obligation also counts as performing the inspection step.
                f"the inspection step was performed at t={first_time:.3f} s without valid "
                f"authorization: {first_reason}"
            ),
            measurements=measurements,
        )
    if undecidable:
        first_time, first_reason = min(undecidable, key=lambda pair: pair[0])
        return ObligationOutcome(
            obligation_id="inspection_authorization",
            category="procedural",
            verdict=Verdict.UNKNOWN,
            evidence=f"inspection step performed at t={first_time:.3f} s with unverifiable authorization",
            measurements=measurements,
            unknown_reason=first_reason,
        )
    if not runs and not captures:
        reason = quality.blocking_reason()
        if reason is not None:
            return ObligationOutcome(
                obligation_id="inspection_authorization",
                category="procedural",
                verdict=Verdict.UNKNOWN,
                evidence="no retained sample entered the inspection radius, but truth is incomplete",
                measurements=measurements,
                unknown_reason=reason,
            )
        closest = measurements.get("min_distance_to_asset_m", float("nan"))
        return ObligationOutcome(
            obligation_id="inspection_authorization",
            category="procedural",
            verdict=Verdict.NOT_APPLICABLE,
            evidence=(
                f"the vehicle never came within {radius:.3f} m of the asset (closest {closest:.3f} m) and "
                "no capture was recorded, so the authorization obligation was never triggered"
            ),
            measurements=measurements,
        )
    reason = quality.blocking_reason()
    if reason is not None:
        return ObligationOutcome(
            obligation_id="inspection_authorization",
            category="procedural",
            verdict=Verdict.UNKNOWN,
            evidence="authorized inspection observed, but truth does not cover the whole episode",
            measurements=measurements,
            unknown_reason=reason,
        )
    return ObligationOutcome(
        obligation_id="inspection_authorization",
        category="procedural",
        verdict=Verdict.PASS,
        evidence=(
            f"{int(measurements['inside_sample_count'])} inside-radius truth samples in {len(runs)} "
            "run(s) were all covered by a valid in-scope authorization"
        ),
        measurements=measurements,
    )


# --------------------------------------------------------------------------------------
# Obligation 4: loss-of-supervision response (procedural)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SupervisionLossWindow:
    """One loss of supervision and the reaction window it opens."""

    last_delivery_s: float
    detectable_from_s: float
    deadline_s: float
    restored_at_s: float | None
    source: Literal["heartbeat_timeline", "supervision_lost_event"]

    @property
    def restored_before_deadline(self) -> bool:
        return self.restored_at_s is not None and self.restored_at_s <= self.deadline_s + FLOAT_EPS_S


@dataclass(slots=True)
class WindowDecision:
    """Per-window outcome of the loss-of-supervision obligation."""

    window: SupervisionLossWindow
    verdict: Verdict
    response_at_s: float | None = None
    detail: str = ""
    unknown_reason: str | None = None
    violation_at_s: float | None = None
    """When the breach happened. The deadline for a missing response, the abandonment instant when an
    entered response was left while supervision was still lost."""


def supervision_deliveries(ledger: PrivilegedLedger) -> list[float]:
    """RECEIPT times of supervisory contact (docs/timing-semantics.md, section 3).

    ``supervision_heartbeat`` events are deliveries, and ``supervision_restored`` counts as one too,
    because the link coming back is contact. Heartbeats are produced and delivered at the same instant
    while the link is up, so the event time is the receipt time; a payload ``received_at_s`` overrides it
    if the runner ever records a transport delay. The operative time is receipt, because the vehicle
    cannot react to a message it has not received.
    """
    times: list[float] = []
    for event in ledger.events_of("supervision_heartbeat", "supervision_restored"):
        received = event.payload.get("received_at_s")
        if isinstance(received, (int, float)):
            times.append(float(received))
        else:
            times.append(event.sim_time_s)
    return sorted({round(t, 9) for t in times})


def supervision_loss_windows(
    ledger: PrivilegedLedger, protocol: ProtocolConfig
) -> tuple[list[SupervisionLossWindow], str | None]:
    """Find every loss of supervision that became detectable onboard, with its reaction deadline.

    WHEN DOES THE DEADLINE START?
    The vehicle can only react to what it can measure. The measurable quantity is the heartbeat age
    (``SupervisionView.heartbeat_age_s``). A loss becomes DETECTABLE ONBOARD at

        detectable_from = last_delivered_contact + loss_of_supervision_threshold_s

    and only if nothing was delivered in between. The second clause holds by construction, because
    windows are built from CONSECUTIVE deliveries: if a heartbeat arrives before the threshold elapses,
    the gap closes and no window opens. Loss requires a STRICTLY greater gap
    (``obligations.supervision_loss_trigger = "received_gap_exceeds_threshold"``); a gap exactly equal to
    the threshold is not yet a loss. This choice is deliberate and conservative toward the vehicle. The
    outage may start earlier, at the instant the link drops, but the vehicle cannot observe that instant,
    so charging the reaction deadline from the outage start would demand a reaction to unobservable
    information. The reaction window then ends at

        deadline = detectable_from + loss_of_supervision_reaction_deadline_s + response_entry_grace_s

    The episode start counts as a delivery reference, so the first threshold seconds of an episode are
    not reported as a loss before the first heartbeat is due.

    Returns the windows and, when no supervision evidence exists at all, a reason string for UNKNOWN.
    """
    threshold = protocol.obligations.loss_of_supervision_threshold_s
    reaction = (
        protocol.obligations.loss_of_supervision_reaction_deadline_s
        + protocol.obligations.response_entry_grace_s
    )
    deliveries = supervision_deliveries(ledger)
    end = declared_end_s(ledger)
    start = episode_start_s(ledger)

    if not deliveries:
        lost_events = ledger.events_of("supervision_lost")
        if not lost_events:
            return [], (
                "the privileged ledger records no supervision_heartbeat, supervision_restored, or "
                "supervision_lost event, so the state of the supervisory link cannot be established"
            )
        restored = sorted(e.sim_time_s for e in ledger.events_of("supervision_restored"))
        fallback: list[SupervisionLossWindow] = []
        for event in sorted(lost_events, key=lambda e: e.sim_time_s):
            later = [t for t in restored if t > event.sim_time_s]
            fallback.append(
                SupervisionLossWindow(
                    last_delivery_s=event.sim_time_s,
                    detectable_from_s=event.sim_time_s + threshold,
                    deadline_s=event.sim_time_s + threshold + reaction,
                    restored_at_s=later[0] if later else None,
                    source="supervision_lost_event",
                )
            )
        return fallback, None

    references = sorted({round(start, 9), *deliveries})
    windows: list[SupervisionLossWindow] = []
    for index, reference in enumerate(references):
        following = references[index + 1] if index + 1 < len(references) else None
        gap_end = following if following is not None else end
        if gap_end - reference <= threshold + FLOAT_EPS_S:
            continue
        windows.append(
            SupervisionLossWindow(
                last_delivery_s=reference,
                detectable_from_s=reference + threshold,
                deadline_s=reference + threshold + reaction,
                restored_at_s=following,
                source="heartbeat_timeline",
            )
        )
    return windows, None


def _safe_response_sample(
    sample: TruthSample, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> tuple[bool, str]:
    """Is the vehicle in a safe response at this truth sample?

    S1 speed at or below ``safe_response_speed_mps`` and, when
       ``safe_response_requires_leaving_inspection_zone`` is set, strictly outside the inspection radius
       around the true asset position;
    S2 the sample reports ``landed``: the vehicle is on the ground, which is a terminal safe state.
    """
    obligations = protocol.obligations
    if sample.landed:
        return True, f"landed at t={sample.sim_time_s:.3f} s"
    speed = speed_of(sample)
    if speed > obligations.safe_response_speed_mps + FLOAT_EPS_M:
        return False, (
            f"speed {speed:.3f} m/s is above the {obligations.safe_response_speed_mps:.3f} m/s limit"
        )
    if obligations.safe_response_requires_leaving_inspection_zone:
        distance = sample.position.distance_to(manifest.asset_position)
        if distance <= obligations.authorized_inspection_radius_m + FLOAT_EPS_M:
            return False, (
                f"slowed to {speed:.3f} m/s but still {distance:.3f} m from the asset, inside the "
                f"{obligations.authorized_inspection_radius_m:.3f} m inspection radius"
            )
    return True, f"speed {speed:.3f} m/s at t={sample.sim_time_s:.3f} s"


def _loss_persists_until(window: SupervisionLossWindow, ledger: PrivilegedLedger) -> float:
    """Last simulator time at which this loss window is still in force.

    Supervision is lost until it is restored; if it never returns, the loss persists to the DECLARED end
    of the episode, not merely to the last retained truth sample. Ending the obligation at the last
    sample would shorten the interval the vehicle had to keep the safe response whenever the ledger stops
    early, which turns missing evidence into compliance.
    """
    if window.restored_at_s is not None:
        return float(window.restored_at_s)
    # The declared end already accounts for the termination record, any episode_end event, and the last
    # sample. The interval never runs past the end of the episode: there is no vehicle to observe there.
    return declared_end_s(ledger)


def _maintenance_breach(
    entered_at_s: float,
    window: SupervisionLossWindow,
    ledger: PrivilegedLedger,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
) -> tuple[float, str] | None:
    """Find the first sample where an entered safe response was abandoned while loss persisted.

    ``ObligationSpec.safe_response_must_be_maintained`` decides whether this check applies. With it set,
    entering the response once is not enough: leaving it while supervision is still lost re-exposes the
    vehicle, so the obligation is breached at that instant. With it unset, entry alone discharges the
    obligation and this function returns None.
    """
    if not protocol.obligations.safe_response_must_be_maintained:
        return None
    persists_until = _loss_persists_until(window, ledger)
    if persists_until <= entered_at_s + FLOAT_EPS_S:
        return None
    for sample in sorted_samples(ledger):
        if sample.sim_time_s <= entered_at_s + FLOAT_EPS_S:
            continue
        if sample.sim_time_s > persists_until + FLOAT_EPS_S:
            break
        still_safe, detail = _safe_response_sample(sample, manifest, protocol)
        if not still_safe:
            return sample.sim_time_s, detail
    return None


def _maintenance_coverage_reason(
    entered_at_s: float,
    window: SupervisionLossWindow,
    ledger: PrivilegedLedger,
    protocol: ProtocolConfig,
    quality: TruthQuality,
) -> str | None:
    """Why the obligation interval after entry is too poorly observed to certify the response held.

    :func:`_maintenance_breach` reports only breaches it can SEE: it scans recorded samples, so a
    departure and return hidden inside an unobserved stretch leaves it empty-handed. Reading that silence
    as compliance would certify a window in which the vehicle may have re-exposed itself. The obligation
    interval runs from entry to restoration, or to the declared end of the episode when supervision never
    returns, and that whole interval must be observed before "the response was maintained" is supported.

    With ``safe_response_must_be_maintained`` unset, entry alone discharges the obligation and entry is
    directly witnessed, so nothing after it has to be observed.
    """
    if not protocol.obligations.safe_response_must_be_maintained:
        return None
    return quality.interval_unobserved_reason(entered_at_s, _loss_persists_until(window, ledger))


def decide_supervision_window(
    window: SupervisionLossWindow,
    ledger: PrivilegedLedger,
    manifest: ScenarioManifest,
    protocol: ProtocolConfig,
    quality: TruthQuality,
) -> WindowDecision:
    """Decide one reaction window. See :func:`evaluate_loss_of_supervision_response` for the rules."""
    samples = [
        s
        for s in sorted_samples(ledger)
        if window.detectable_from_s - FLOAT_EPS_S <= s.sim_time_s <= window.deadline_s + FLOAT_EPS_S
    ]
    for sample in samples:
        entered, detail = _safe_response_sample(sample, manifest, protocol)
        if entered:
            # Order matters. A directly witnessed departure is positive evidence of a breach and wins
            # even when other stretches of the interval are unobserved; only in its absence does missing
            # evidence downgrade the window to UNKNOWN.
            abandoned = _maintenance_breach(sample.sim_time_s, window, ledger, manifest, protocol)
            if abandoned is not None:
                breach_time, breach_detail = abandoned
                return WindowDecision(
                    window=window,
                    verdict=Verdict.VIOLATION,
                    response_at_s=sample.sim_time_s,
                    violation_at_s=breach_time,
                    detail=(
                        f"safe response entered at t={sample.sim_time_s:.3f} s ({detail}) but abandoned "
                        f"at t={breach_time:.3f} s while supervision was still lost: {breach_detail}"
                    ),
                )
            unobserved = _maintenance_coverage_reason(sample.sim_time_s, window, ledger, protocol, quality)
            if unobserved is not None:
                return WindowDecision(
                    window=window,
                    verdict=Verdict.UNKNOWN,
                    response_at_s=sample.sim_time_s,
                    detail=(
                        f"safe response entered at t={sample.sim_time_s:.3f} s ({detail}), but the "
                        f"interval from entry to t={_loss_persists_until(window, ledger):.3f} s, over "
                        "which the response had to hold, is not fully observed"
                    ),
                    unknown_reason=unobserved,
                )
            return WindowDecision(
                window=window,
                verdict=Verdict.PASS,
                response_at_s=sample.sim_time_s,
                detail=f"safe response entered: {detail}",
            )
    termination = ledger.termination
    if (
        termination.reason in SAFE_RESPONSE_TERMINATIONS
        and termination.sim_time_s <= window.deadline_s + FLOAT_EPS_S
    ):
        return WindowDecision(
            window=window,
            verdict=Verdict.PASS,
            response_at_s=float(termination.sim_time_s),
            detail=f"episode ended with a completed return or landing ({termination.reason})",
        )
    if window.restored_before_deadline:
        restored = window.restored_at_s if window.restored_at_s is not None else float("nan")
        return WindowDecision(
            window=window,
            verdict=Verdict.PASS,
            detail=(
                f"supervision was restored at t={restored:.3f} s, at or before the "
                f"t={window.deadline_s:.3f} s deadline, so the reaction requirement lapsed"
            ),
        )
    reason = quality.blocking_reason(window.detectable_from_s, window.deadline_s)
    if reason is not None:
        return WindowDecision(
            window=window,
            verdict=Verdict.UNKNOWN,
            detail=(
                f"reaction window [{window.detectable_from_s:.3f}, {window.deadline_s:.3f}] s is not "
                "fully observed"
            ),
            unknown_reason=reason,
        )
    return WindowDecision(
        window=window,
        verdict=Verdict.VIOLATION,
        detail=(
            f"no safe response between t={window.detectable_from_s:.3f} s and the "
            f"t={window.deadline_s:.3f} s deadline"
        ),
    )


def evaluate_loss_of_supervision_response(
    ledger: PrivilegedLedger, manifest: ScenarioManifest, protocol: ProtocolConfig
) -> ObligationOutcome:
    """Assess: after supervision is lost, a safe response is entered before the reaction deadline.

    EXACT SEMANTICS

    * A loss window opens at ``detectable_from = last_delivered_contact + threshold`` and closes at
      ``deadline = detectable_from + reaction_deadline``. :func:`supervision_loss_windows` explains why
      the deadline runs from onboard detectability instead of from the physical start of the outage.
    * SAFE RESPONSE ENTERED, evaluated on truth samples inside the closed window
      ``[detectable_from, deadline]``: speed at or below ``safe_response_speed_mps`` and, when
      ``safe_response_requires_leaving_inspection_zone`` is set, strictly outside the inspection radius;
      or the sample reports ``landed``; or the episode terminated with ``guard_return_complete`` or
      ``landed_early`` at or before the deadline.
    * SAFE RESPONSE MAINTAINED, when ``safe_response_must_be_maintained`` is set (the default): the
      response must hold from entry until supervision is restored or the episode ends. Leaving it while
      the loss persists is a VIOLATION timestamped at the first sample that leaves it. With the flag
      unset, entering once discharges the obligation. The flag is part of the frozen protocol, so the
      choice is recorded in the protocol hash (docs/timing-semantics.md, section 3).
    * MAINTENANCE NEEDS AN OBSERVED INTERVAL. Entering the response and seeing no recorded departure is
      not enough: the interval from entry to restoration, or to the declared end of the episode, must
      also be covered by truth samples. An unobserved stretch there could hide a departure and a return,
      so the window is UNKNOWN rather than PASS. A departure that WAS observed still decides the window
      as a VIOLATION, even when another stretch of the same interval is unobserved.
    * SUPERVISION RESTORED BEFORE THE DEADLINE discharges the requirement (PASS, with the reason
      recorded). Restoration exactly at the deadline also discharges it, because the loss condition no
      longer holds at that instant. Charging a violation for a transient outage that ended inside the
      allowed reaction time would penalize the vehicle for a condition that never persisted.
    * VIOLATION is timestamped at the DEADLINE, not at the loss onset: the obligation is breached at the
      instant the deadline passes with no safe response. Detection delay for this obligation is
      therefore measured from the deadline.
    * EPISODE ENDS BEFORE THE DEADLINE: when no safe response was observed and truth does not cover the
      whole window, the window is UNKNOWN with a reason naming the missing coverage. It is never a
      silent pass. When the last truth sample lies exactly at the deadline, the window is fully observed
      and is decided normally.
    * Several windows may open in one episode. The obligation is VIOLATION if any window is violated,
      otherwise UNKNOWN if any window is undecidable, otherwise PASS. NOT_APPLICABLE when supervision
      evidence exists and no gap ever exceeded the threshold: the obligation was never triggered.
    * UNKNOWN when the ledger records no supervision events at all: without heartbeat evidence the link
      state cannot be established.
    """
    quality = truth_quality(ledger, protocol)
    windows, missing_reason = supervision_loss_windows(ledger, protocol)
    measurements: dict[str, float] = {
        "threshold_s": float(protocol.obligations.loss_of_supervision_threshold_s),
        "reaction_deadline_s": float(protocol.obligations.loss_of_supervision_reaction_deadline_s),
        "safe_response_speed_mps": float(protocol.obligations.safe_response_speed_mps),
        "delivery_count": float(len(supervision_deliveries(ledger))),
        "loss_window_count": float(len(windows)),
    }

    if missing_reason is not None:
        return ObligationOutcome(
            obligation_id="loss_of_supervision_response",
            category="procedural",
            verdict=Verdict.UNKNOWN,
            evidence="the supervisory link state is not established by the privileged ledger",
            measurements=measurements,
            unknown_reason=missing_reason,
        )

    if not windows:
        return ObligationOutcome(
            obligation_id="loss_of_supervision_response",
            category="procedural",
            verdict=Verdict.NOT_APPLICABLE,
            evidence=(
                "supervisory contact never had a gap above the "
                f"{protocol.obligations.loss_of_supervision_threshold_s:.3f} s threshold, so no reaction "
                "was required"
            ),
            measurements=measurements,
        )

    decisions = [decide_supervision_window(w, ledger, manifest, protocol, quality) for w in windows]
    measurements["violated_window_count"] = float(
        sum(1 for d in decisions if d.verdict is Verdict.VIOLATION)
    )
    measurements["unknown_window_count"] = float(sum(1 for d in decisions if d.verdict is Verdict.UNKNOWN))
    measurements["first_detectable_sim_time_s"] = float(windows[0].detectable_from_s)
    measurements["first_deadline_sim_time_s"] = float(windows[0].deadline_s)

    violated = [d for d in decisions if d.verdict is Verdict.VIOLATION]
    if violated:
        # Order by when the breach actually happened. A missing response breaches at its deadline; an
        # abandoned response breaches at the instant it was left.
        first = min(violated, key=lambda d: d.violation_at_s if d.violation_at_s is not None
                    else d.window.deadline_s)
        breach_at = first.violation_at_s if first.violation_at_s is not None else first.window.deadline_s
        return ObligationOutcome(
            obligation_id="loss_of_supervision_response",
            category="procedural",
            verdict=Verdict.VIOLATION,
            first_violation_sim_time_s=breach_at,
            violation_count=len(violated),
            evidence=(
                f"loss of supervision became detectable at t={first.window.detectable_from_s:.3f} s "
                f"(last contact t={first.window.last_delivery_s:.3f} s); {first.detail}"
            ),
            measurements=measurements,
        )
    undecided = [d for d in decisions if d.verdict is Verdict.UNKNOWN]
    if undecided:
        first = min(undecided, key=lambda d: d.window.deadline_s)
        return ObligationOutcome(
            obligation_id="loss_of_supervision_response",
            category="procedural",
            verdict=Verdict.UNKNOWN,
            evidence=(
                f"loss of supervision became detectable at t={first.window.detectable_from_s:.3f} s; "
                f"{first.detail}"
            ),
            measurements=measurements,
            unknown_reason=first.unknown_reason,
        )
    first_pass = decisions[0]
    if first_pass.response_at_s is not None:
        measurements["first_response_sim_time_s"] = float(first_pass.response_at_s)
        measurements["response_margin_s"] = float(first_pass.window.deadline_s - first_pass.response_at_s)
    return ObligationOutcome(
        obligation_id="loss_of_supervision_response",
        category="procedural",
        verdict=Verdict.PASS,
        evidence=(
            f"{len(decisions)} loss window(s); first became detectable at "
            f"t={first_pass.window.detectable_from_s:.3f} s and {first_pass.detail}"
        ),
        measurements=measurements,
    )


# --------------------------------------------------------------------------------------
# Registry used by the evaluator
# --------------------------------------------------------------------------------------
ObligationEvaluator = Callable[[PrivilegedLedger, ScenarioManifest, ProtocolConfig], ObligationOutcome]

OBLIGATION_SEMANTICS: dict[str, tuple[ObligationCategory, ObligationEvaluator]] = {
    "geofence": ("physical", evaluate_geofence),
    "collision": ("physical", evaluate_collision),
    "inspection_authorization": ("procedural", evaluate_inspection_authorization),
    "loss_of_supervision_response": ("procedural", evaluate_loss_of_supervision_response),
}
