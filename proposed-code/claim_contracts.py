"""Post-study reporting reference. No simulator connection or control authority.

Synthetic software tests do not validate safety in the experiment or real world.
Times denote sampled evidence; no interpolation or continuous guarantee is made.
"""
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class Verdict(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ClaimContract:
    obligation: str
    start: float
    end: float
    denominator: str
    affected_parties: tuple[str, ...]
    admissible_evidence: str
    normative_justification: str | None = None
    challenge_procedure: str | None = None

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.start, self.end)) or self.end < self.start:
            raise ValueError("Horizon must be finite and ordered")
        if not all((self.obligation, self.denominator, self.affected_parties, self.admissible_evidence)):
            raise ValueError("Identify obligation, denominator, parties and evidence")

    @property
    def normative_fields_documented(self):
        """Presence is not validity, legitimacy or stakeholder agreement."""
        return bool(self.normative_justification and self.challenge_procedure)


def conformance(*, known_violation: bool, coverage_complete: bool,
                all_required_evidence_valid: bool) -> Verdict:
    """An established violation survives missing evidence elsewhere."""
    if known_violation:
        return Verdict.REFUTED
    if coverage_complete and all_required_evidence_valid:
        return Verdict.SUPPORTED
    return Verdict.UNRESOLVED


def false_assurance_bounds(*, accepted: int, known_violation: int, unresolved: int):
    """Bounds conditional on accepted cases, not causal guard comparisons."""
    values = (accepted, known_violation, unresolved)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Counts must be nonnegative integers")
    if known_violation + unresolved > accepted:
        raise ValueError("Disjoint violation and unresolved counts exceed acceptance")
    if accepted == 0:
        return None
    return (known_violation / accepted, (known_violation + unresolved) / accepted)


def sampled_fallback(contract: ClaimContract, samples: list[tuple[float, bool | None]],
                     max_gap: float) -> Verdict:
    """Assess ONLY sampled conformance over exactly the requested horizon.

    Missing endpoints, excessive gaps, unknown samples, empty windows, or a
    non-increasing clock with no observed violation cannot support conformance.
    A False predicate means a validly established violation, not a sensor error.
    Evidence validity and authority are upstream requirements, not inferred here.
    """
    if not isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    if any(not isfinite(t) or (v is not None and type(v) is not bool) for t, v in samples):
        raise ValueError("Samples require finite times and Boolean or unknown values")
    relevant = [(t, v) for t, v in samples if contract.start <= t <= contract.end]
    if any(v is False for _, v in relevant):
        return Verdict.REFUTED
    if not relevant:
        return Verdict.UNRESOLVED
    times = [t for t, _ in relevant]
    complete = (times[0] == contract.start and times[-1] == contract.end
                and all(0 < b - a <= max_gap for a, b in zip(times, times[1:])))
    valid = all(v is True for _, v in relevant)
    return conformance(known_violation=False, coverage_complete=complete,
                       all_required_evidence_valid=valid)


def may_reuse_support(source: ClaimContract, target: ClaimContract) -> bool:
    """Conservative identity gate; caller must still check actual evidence.

    Even shortening a horizon can change eligibility or the decision denominator,
    so this example permits exact contract reuse only. True is NOT assurance.
    """
    return source == target
