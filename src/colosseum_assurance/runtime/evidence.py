"""Evidence writing with hard separation between exposed records and privileged truth.

Layout under ``results/<run_class>/<protocol_short_hash>/``::

    run_metadata.json          run class, protocol hash, code version, simulator provenance
    manifests/<scenario>.json  scenario realizations (shared by all arms)
    episodes/<episode>.json    exposed EpisodeRecord: what controller and monitor could see and did
    privileged_ledgers/<episode>.json  evaluator-only TruthSample/TruthEvent ledger
    frames/<episode>/...       saved RGB/depth frames for replay and inspection
    attempted_runs.jsonl       every attempt, including crashes and timeouts

Two safeguards live here, because mixing evidence classes is the easiest way to ruin the study:
``ProvenanceError`` blocks fixture data from a pilot or held-out tree, and ``RunTreeConflict`` blocks two
different protocol hashes or run classes from sharing one tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colosseum_assurance.config import PathsConfig
from colosseum_assurance.scenario.manifest import ScenarioManifest
from colosseum_assurance.schemas import (
    ANCHORED_LIVE_PROVENANCES,
    AttemptedRun,
    EpisodeRecord,
    PrivilegedLedger,
    SimulatorIdentity,
)
from colosseum_assurance.version import code_version

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from colosseum_assurance.protocol.spec import ProtocolConfig

EXPERIMENTAL_RUN_CLASSES = frozenset({"pilot", "heldout"})


class ProvenanceError(RuntimeError):
    """Raised when fixture-fake evidence is about to be written into an experimental run tree."""


class RunTreeConflict(RuntimeError):
    """Raised when a run directory already holds a different protocol hash or run class."""


class EvidenceExists(RunTreeConflict):
    """Raised when writing would replace evidence that already exists.

    A retry after a failure used to overwrite the earlier episode, its ledger, and its frames while
    appending a second attempt to the ledger. The run then looked like two attempts with one visible
    outcome, and the failed evidence was gone. Nothing may overwrite recorded evidence.
    """


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def assert_provenance_allowed(identity: SimulatorIdentity, run_class: str) -> None:
    """Refuse to treat fixture or unverified simulator output as experimental evidence."""
    if run_class in EXPERIMENTAL_RUN_CLASSES and not identity.is_live:
        raise ProvenanceError(
            f"run_class={run_class!r} requires a live Colosseum simulator, but simulator provenance is "
            f"{identity.provenance!r} (endpoint {identity.endpoint_label!r}). Fixture output is for "
            "software tests only and can never be experimental evidence."
        )


@dataclass
class EvidenceWriter:
    """Creates the run tree and writes every artifact of one run class.

    ``protocol`` is optional only so existing callers keep working. Supplying it persists the exact
    protocol into the run tree, which is what lets post-processing score the run under the protocol it
    actually ran with instead of under current defaults.
    """

    paths: PathsConfig
    run_class: str
    protocol_hash: str
    protocol: Any = None

    def __post_init__(self) -> None:
        self.protocol_short_hash = self.protocol_hash.split(":", 1)[-1][:12]
        self.root = self.paths.run_dir(self.run_class, self.protocol_short_hash)
        self.episodes = self.root / "episodes"
        self.ledgers = self.root / "privileged_ledgers"
        self.manifests = self.root / "manifests"
        self.frames = self.root / self.paths.frames_subdir
        for d in (self.episodes, self.ledgers, self.manifests, self.frames):
            d.mkdir(parents=True, exist_ok=True)
        self._init_metadata()
        if self.protocol is not None:
            if self.protocol.content_hash() != self.protocol_hash:
                raise RunTreeConflict(
                    f"the supplied protocol hashes to {self.protocol.content_hash()} but this tree is "
                    f"{self.protocol_hash}"
                )
            write_run_protocol(self.root, self.protocol, {"run_class": self.run_class})
            extension = self.protocol.study_extension
            if extension is not None and extension.perception_split_by_group:
                inventory = {"groups": extension.perception_split_by_group,
                             "protocol_hash": self.protocol_hash, "run_class": self.run_class}
                path = self.root / "perception" / "group-inventory.json"
                if path.exists() and json.loads(path.read_text()) != inventory:
                    raise RunTreeConflict("frozen perception group inventory changed")
                if not path.exists():
                    self._write_json(path, inventory)

    # ---------------------------------------------------------------- metadata
    @property
    def metadata_path(self) -> Path:
        return self.root / "run_metadata.json"

    def _init_metadata(self) -> None:
        meta = {
            "run_class": self.run_class,
            "protocol_hash": self.protocol_hash,
            "protocol_short_hash": self.protocol_short_hash,
            "created_utc": utc_now(),
            "code_version": code_version(),
            "schema_note": "see docs/data-schemas.md",
        }
        if self.metadata_path.exists():
            existing = json.loads(self.metadata_path.read_text())
            if existing.get("protocol_hash") != self.protocol_hash:
                raise RunTreeConflict(
                    f"{self.metadata_path} already belongs to protocol {existing.get('protocol_hash')!r}; "
                    f"refusing to add {self.protocol_hash!r}. Use a separate results tree."
                )
            if existing.get("run_class") != self.run_class:
                raise RunTreeConflict(
                    f"{self.metadata_path} already belongs to run_class {existing.get('run_class')!r}; "
                    f"refusing to mix with {self.run_class!r}."
                )
            return
        self._write_json(self.metadata_path, meta)

    def note_simulator(self, identity: SimulatorIdentity) -> None:
        """Record which simulator produced this run tree, and refuse a provenance downgrade."""
        assert_provenance_allowed(identity, self.run_class)
        meta = json.loads(self.metadata_path.read_text())
        seen = meta.setdefault("simulator_identities", [])
        entry = identity.model_dump(mode="json")
        if entry not in seen:
            seen.append(entry)
        provenances = {str(e.get("provenance")) for e in seen}
        meta["simulator_provenance"] = sorted(provenances)
        if self.run_class in EXPERIMENTAL_RUN_CLASSES and provenances - set(ANCHORED_LIVE_PROVENANCES):
            raise ProvenanceError(
                f"run tree {self.root} would mix simulator provenances {sorted(provenances)}; experimental "
                f"runs accept anchored provenance only: {sorted(ANCHORED_LIVE_PROVENANCES)}."
            )
        meta["updated_utc"] = utc_now()
        self._write_json(self.metadata_path, meta)

    # ------------------------------------------------------------------ writes
    def write_manifest(self, manifest: ScenarioManifest) -> Path:
        """Write the manifest that was actually flown.

        Arms of one scenario share it, so re-writing an identical manifest is fine. A DIFFERENT manifest
        under the same scenario id would mean the arms did not fly matched geometry, which is refused.
        """
        path = self.manifests / f"{manifest.scenario_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != manifest.model_dump(mode="json"):
                raise EvidenceExists(
                    f"{path} already holds a different manifest for scenario {manifest.scenario_id!r}. "
                    "Arms of one scenario must fly the same geometry, so the matched set is broken."
                )
            return path
        self._write_json(path, manifest.model_dump(mode="json"))
        return path

    def record_scenario_deviation(self, payload: dict[str, Any]) -> Path:
        """Append a scenario deviation, for example geometry bound to a qualified third-party map."""
        path = self.root / "scenario_deviations.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"recorded_utc": utc_now(), **payload}, sort_keys=True) + "\n")
        return path

    def existing_evidence(self, episode_id: str) -> dict[str, str]:
        """Every artifact this run tree already holds for that episode id.

        A crashed run can leave a ledger, a frames directory, or an attempt row without an episode
        file. Checking only the episode file would let a repeat overwrite that partial evidence, which
        is exactly what the attempted-run ledger exists to prevent. The keys are artifact kinds and the
        values are what was found.
        """
        found: dict[str, str] = {}
        episode = self.episodes / f"{episode_id}.json"
        if episode.exists():
            found["episode_record"] = str(episode)
        ledger = self.ledgers / f"{episode_id}.json"
        if ledger.exists():
            found["privileged_ledger"] = str(ledger)
        pairs = self.root / "privileged_state_transformations" / f"{episode_id}.jsonl"
        if pairs.exists():
            found["privileged_state_transformations"] = str(pairs)
        camera_pairs = self.root / "privileged_camera_transformations" / f"{episode_id}.jsonl"
        if camera_pairs.exists():
            found["privileged_camera_transformations"] = str(camera_pairs)
        frames = self.frames / episode_id
        if frames.exists() and any(frames.iterdir()):
            found["frames"] = str(frames)
        attempts_path = self.paths.attempted_runs_path(self.run_class, self.protocol_short_hash)
        if attempts_path.exists():
            rows = [
                line for line in attempts_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("episode_id") == episode_id
            ]
            if rows:
                found["attempted_runs"] = f"{len(rows)} row(s) in {attempts_path}"
        return found

    def episode_exists(self, episode_id: str) -> bool:
        """True when this run tree already holds ANY evidence for that episode id."""
        return bool(self.existing_evidence(episode_id))

    def assert_episode_not_recorded(self, episode_id: str) -> None:
        """Refuse a repeat before the simulator is touched.

        Called by the runner before reset, so a duplicate scenario and arm cannot mutate the world,
        append a second attempt row, and overwrite evidence from the first try. The whole inventory is
        checked, including a ledger, frames, or an attempt row left behind by a run that failed before
        it wrote its episode file.
        """
        found = self.existing_evidence(episode_id)
        if found:
            inventory = "; ".join(f"{kind}: {where}" for kind, where in sorted(found.items()))
            raise EvidenceExists(
                f"episode {episode_id!r} already has recorded evidence in {self.root} ({inventory}). "
                "Recorded evidence is never replaced, including the partial evidence of a failed run. "
                "Use a fresh results tree, or resume the run so already-recorded episodes are skipped."
            )

    def write_episode(self, record: EpisodeRecord) -> Path:
        assert_provenance_allowed(record.simulator_identity, record.run_class)
        if record.protocol_hash != self.protocol_hash:
            raise RunTreeConflict(
                f"episode {record.episode_id} carries protocol {record.protocol_hash!r}, tree is "
                f"{self.protocol_hash!r}"
            )
        path = self.episodes / f"{record.episode_id}.json"
        if path.exists():
            raise EvidenceExists(
                f"{path} already exists; refusing to overwrite recorded evidence for episode "
                f"{record.episode_id!r}"
            )
        self._write_json(path, record.model_dump(mode="json"))
        return path

    def write_ledger(self, ledger: PrivilegedLedger) -> Path:
        assert_provenance_allowed(ledger.simulator_identity, ledger.run_class)
        path = self.ledgers / f"{ledger.episode_id}.json"
        if path.exists():
            raise EvidenceExists(
                f"{path} already exists; refusing to overwrite the privileged ledger of episode "
                f"{ledger.episode_id!r}"
            )
        self._write_json(path, ledger.model_dump(mode="json"))
        return path

    def frame_prefix(self, episode_id: str, step_index: int) -> Path:
        d = self.frames / episode_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"step{step_index:04d}"

    def append_attempt(self, attempt: AttemptedRun) -> Path:
        path = self.paths.attempted_runs_path(self.run_class, self.protocol_short_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(attempt.model_dump(mode="json"), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return path

    def append_state_transformation(self, episode_id: str, payload: dict[str, Any]) -> Path:
        """Evaluator-only pre/post injection evidence; never attached to exposed observations."""
        if Path(episode_id).name != episode_id or episode_id in {"", ".", ".."}:
            raise ValueError("invalid episode ID for state transformation evidence")
        row = dict(payload, episode_id=episode_id, protocol_hash=self.protocol_hash,
                   schema_version="onboard_state_transformation_v1")
        for kind in ("raw_state", "delivered_state"):
            raw = json.dumps(row[kind], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            row[kind + "_sha256"] = hashlib.sha256(raw).hexdigest()
        path = self.root / "privileged_state_transformations" / f"{episode_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)


def load_attempted_runs(path: Path) -> list[AttemptedRun]:
    """Read the attempted-run ledger. Malformed lines raise: silent loss is not acceptable."""
    if not path.exists():
        return []
    out: list[AttemptedRun] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(AttemptedRun.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - we want the line number in the message
            raise ValueError(f"{path}:{i} is not a valid attempted-run record: {exc}") from exc
    return out


def summarize_attempts(attempts: list[AttemptedRun]) -> dict[str, Any]:
    """Counts by status and arm, so partial and failed runs stay visible in every report."""
    by_status: dict[str, int] = {}
    by_arm: dict[str, dict[str, int]] = {}
    for a in attempts:
        by_status[a.status] = by_status.get(a.status, 0) + 1
        by_arm.setdefault(a.arm_id, {})
        by_arm[a.arm_id][a.status] = by_arm[a.arm_id].get(a.status, 0) + 1
    completed = by_status.get("completed", 0)
    return {
        "attempts": len(attempts),
        "by_status": dict(sorted(by_status.items())),
        "by_arm": {k: dict(sorted(v.items())) for k, v in sorted(by_arm.items())},
        "completed": completed,
        "not_completed": len(attempts) - completed,
        "provenances": sorted({a.simulator_provenance for a in attempts}),
    }


# --------------------------------------------------------------------------------------
# Protocol persistence: post-processing must score a run under the protocol that produced it
# --------------------------------------------------------------------------------------
PROTOCOL_FILENAME = "protocol.json"


class ProtocolMismatch(RuntimeError):
    """Raised when a supplied protocol does not match the one a run was produced under."""


def write_run_protocol(run_dir: Path, protocol: ProtocolConfig, meta: dict[str, Any] | None = None) -> Path:
    """Persist the exact protocol a run was produced under, inside the run tree.

    Without this, ``colassure evaluate`` and ``colassure analyze`` fall back to current defaults and
    silently score a run under a protocol it never ran with. The file is written once; a second run in
    the same tree with a different protocol is a :class:`RunTreeConflict`.
    """
    path = Path(run_dir) / PROTOCOL_FILENAME
    payload = {
        "protocol_hash": protocol.content_hash(),
        "protocol_short_hash": protocol.short_hash,
        "written_utc": utc_now(),
        "code_version": code_version(),
        "source": dict(meta or {}),
        "protocol": protocol.model_dump(mode="json"),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != payload["protocol_hash"]:
            raise RunTreeConflict(
                f"{path} already holds protocol {existing.get('protocol_hash')!r}; refusing to replace it "
                f"with {payload['protocol_hash']!r}"
            )
        return path
    EvidenceWriter._write_json(path, payload)
    return path


def load_run_protocol(
    run_dir: Path, supplied: ProtocolConfig | None = None
) -> tuple[ProtocolConfig, dict[str, Any]]:
    """Load the protocol a run was produced under, and refuse a mismatched supplied protocol.

    Returns ``(protocol, provenance)`` where ``provenance["source"]`` is one of ``run_tree``,
    ``supplied_matching``, ``supplied_unverified``, or ``episode_record_hash_only``.

    There is no defaults fallback. A historical run whose protocol cannot be established raises
    :class:`ProtocolMismatch` with the remedy, because scoring it under current defaults would measure
    it against a specification it never saw. Current defaults are accepted only when they reproduce the
    protocol hash recorded in the run's own evidence.
    """
    from colosseum_assurance.protocol.spec import ProtocolConfig

    run_dir = Path(run_dir)
    path = run_dir / PROTOCOL_FILENAME
    recorded_hash = _recorded_protocol_hash(run_dir)

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = ProtocolConfig.model_validate(payload["protocol"])
        if stored.content_hash() != payload.get("protocol_hash"):
            raise ProtocolMismatch(
                f"{path} was edited after it was written: recorded {payload.get('protocol_hash')}, "
                f"recomputed {stored.content_hash()}"
            )
        if recorded_hash and stored.content_hash() != recorded_hash:
            raise ProtocolMismatch(
                f"{path} holds protocol {stored.content_hash()} but the episodes in {run_dir} carry "
                f"{recorded_hash}"
            )
        if supplied is not None and supplied.content_hash() != stored.content_hash():
            raise ProtocolMismatch(
                f"the supplied protocol ({supplied.content_hash()}) is not the protocol this run was "
                f"produced under ({stored.content_hash()}). Post-processing must use the run's own "
                "protocol, or you are scoring against a specification the run never saw."
            )
        return stored, {
            "source": "run_tree",
            "path": str(path),
            "protocol_hash": stored.content_hash(),
            "written_utc": payload.get("written_utc"),
        }

    if supplied is not None:
        if recorded_hash and supplied.content_hash() != recorded_hash:
            raise ProtocolMismatch(
                f"the supplied protocol ({supplied.content_hash()}) does not match the protocol hash "
                f"recorded in this run's evidence ({recorded_hash})"
            )
        if not recorded_hash:
            return supplied, {
                "source": "supplied_unverified",
                "protocol_hash": supplied.content_hash(),
                "warning": (
                    "this run tree holds neither protocol.json nor a recorded protocol hash, so the "
                    "supplied protocol could not be checked against the run"
                ),
            }
        return supplied, {
            "source": "supplied_matching",
            "protocol_hash": supplied.content_hash(),
            "recorded_protocol_hash": recorded_hash,
            "warning": "the run tree holds no protocol.json; the supplied protocol matched the records",
        }

    # No protocol.json and no supplied protocol. Falling back to current defaults would score a
    # historical run against a specification it never saw, so this refuses instead of warning. The
    # only accepted case is a recorded hash that the current defaults reproduce exactly.
    default = ProtocolConfig()
    if recorded_hash and default.content_hash() == recorded_hash:
        return default, {
            "source": "episode_record_hash_only",
            "protocol_hash": default.content_hash(),
            "recorded_protocol_hash": recorded_hash,
            "note": (
                "this run tree holds no protocol.json, but the current defaults reproduce the protocol "
                "hash recorded in its evidence, so they are the protocol that produced it"
            ),
        }
    if recorded_hash:
        raise ProtocolMismatch(
            f"{run_dir} holds no {PROTOCOL_FILENAME} and its evidence records protocol "
            f"{recorded_hash}, which the current defaults ({default.content_hash()}) do not reproduce. "
            "Re-run with --protocol pointing at the frozen protocol that produced this run; scoring it "
            "under current defaults would measure it against a specification it never saw."
        )
    raise ProtocolMismatch(
        f"{run_dir} holds neither {PROTOCOL_FILENAME} nor a recorded protocol hash, so the protocol "
        "that produced it cannot be established. Supply it with --protocol."
    )


def _recorded_protocol_hash(run_dir: Path) -> str | None:
    """Protocol hash as recorded in the run's own evidence, preferring run metadata."""
    metadata = Path(run_dir) / "run_metadata.json"
    if metadata.exists():
        value = json.loads(metadata.read_text(encoding="utf-8")).get("protocol_hash")
        if isinstance(value, str) and value:
            return value
    for episode in sorted((Path(run_dir) / "episodes").glob("*.json"))[:1]:
        value = json.loads(episode.read_text(encoding="utf-8")).get("protocol_hash")
        if isinstance(value, str) and value:
            return value
    return None
