"""Colosseum Assurance: civilian UAV runtime-assurance study package.

Sub-packages:
  rpc/         minimal msgpack-rpc transport for the Colosseum simulator API
  sim/         Colosseum adapter, simulator identity, scene description, fixture fake server
  scenario/    scenario manifests and precomputed exogenous schedules
  control/     perception path and the fixed goal-directed controller (no privileged state)
  monitors/    runtime guards (policy-only and assumption-aware comparators)
  runtime/     closed-loop episode runner, exposed evidence records, privileged truth ledger, run ledger
  evaluation/  independent evaluator (reads the privileged ledger only)
  audit/       offline audit-record ablation and evidence reconstruction
  analysis/    paired episode-level statistics, metrics, figures, reports
  protocol/    protocol configuration, freezing, and hashing
"""

from colosseum_assurance.version import PACKAGE_VERSION, code_version

__all__ = ["PACKAGE_VERSION", "code_version"]
