"""Audit-evidence reconstruction: what a later evaluator can establish from retained records.

The package is split so the information boundary is visible in the import graph, not only in prose:

* :mod:`~colosseum_assurance.audit.records` applies the prespecified offline field ablations to a
  completed episode while proving the trajectory is untouched.
* :mod:`~colosseum_assurance.audit.reference` derives the answer key from the privileged ledger. It is
  the only module here that may read privileged evidence.
* :mod:`~colosseum_assurance.audit.reconstructor` answers the audit questions from one ablated record
  and the public protocol alone. It imports neither the ledger nor the reference module.
* :mod:`~colosseum_assurance.audit.scoring` compares the two and reports correct, confidently
  incorrect, and insufficient-evidence outcomes with episode-level uncertainty.

Offline record ablation cannot change flight safety; it changes only what an auditor can establish
afterwards.
"""

from __future__ import annotations

from colosseum_assurance.audit.reconstructor import (
    EpisodeReconstruction,
    ReconstructedAnswer,
    reconstruct,
    reconstruct_episode,
)
from colosseum_assurance.audit.records import (
    PROTECTED_STEP_FIELDS,
    AblationError,
    AuditRecordSet,
    build_record_variant,
    build_record_variants,
    build_variants,
    field_paths,
    paths_under,
    removed_paths,
    trajectory_signature,
)
from colosseum_assurance.audit.reference import (
    DESIGNATED_DECISION_STEP_RULE,
    EpisodeReference,
    ReferenceAnswer,
    build_reference,
    reference_answers,
)
from colosseum_assurance.audit.scoring import (
    AUDIT_BOOTSTRAP_RESAMPLES,
    SCORE_CATEGORIES,
    AuditAnalysis,
    AuditScoreSummary,
    EpisodeAuditScore,
    QuestionScore,
    VariantScore,
    aggregate,
    score_episode,
    score_run,
)

__all__ = [
    "AUDIT_BOOTSTRAP_RESAMPLES",
    "DESIGNATED_DECISION_STEP_RULE",
    "PROTECTED_STEP_FIELDS",
    "SCORE_CATEGORIES",
    "AblationError",
    "AuditAnalysis",
    "AuditRecordSet",
    "AuditScoreSummary",
    "EpisodeAuditScore",
    "EpisodeReconstruction",
    "EpisodeReference",
    "QuestionScore",
    "ReconstructedAnswer",
    "ReferenceAnswer",
    "VariantScore",
    "aggregate",
    "build_record_variant",
    "build_record_variants",
    "build_reference",
    "build_variants",
    "field_paths",
    "paths_under",
    "reconstruct",
    "reconstruct_episode",
    "reference_answers",
    "removed_paths",
    "score_episode",
    "score_run",
    "trajectory_signature",
]
