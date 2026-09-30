"""Episode-level analysis: statistics, metrics, deterministic report, and figures.

Nothing in this package produces numbers of its own. It reads ``EpisodeOutcome`` records written by
the independent evaluator and audit scores written by :mod:`colosseum_assurance.audit`, and it refuses
to invent a value where the underlying data does not define one.
"""

from __future__ import annotations

__all__ = ["figures", "metrics", "report", "stats"]
