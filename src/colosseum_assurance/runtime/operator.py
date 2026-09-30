"""Local operator approval queue with explicit provenance and wall-clock timing.

This supplies an interaction surface, not evidence of meaningful human control.
The existing pause/wall-clock simulator settings determine execution pacing.
No network listener or external messaging service is involved.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from colosseum_assurance.runtime.evidence import utc_now
from colosseum_assurance.runtime.supervision import AuthorizationBroker
from colosseum_assurance.schemas import AuthorizationStatus, AuthorizationView, TruthEvent


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _publish(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete file atomically, refusing to replace an existing decision."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic create-if-absent, unlike replace()
    finally:
        temporary.unlink(missing_ok=True)


def submit_response(request_path: Path, *, decision: str, operator: str) -> Path:
    if decision not in {"granted", "denied"} or not operator.strip():
        raise ValueError("decision must be granted/denied and operator label must be nonempty")
    request = json.loads(request_path.read_text())
    if request.get("record_type") != "operator_authorization_request_v1":
        raise ValueError("not an operator request")
    response = {"record_type": "operator_authorization_response_v1", "request_hash": _digest(request),
                "episode_id": request["episode_id"], "token_id": request["token_id"],
                "decision": decision, "operator_label": operator.strip(), "submitted_utc": utc_now()}
    path = request_path.with_name(request_path.name.replace("request-", "response-"))
    _publish(path, response)
    return path


class OperatorQueueBroker(AuthorizationBroker):
    def __init__(self, schedules, obligations, directory: Path, episode_id: str):
        super().__init__(schedules, obligations)
        self.directory = directory / episode_id
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.episode_id = episode_id
        self.pending: dict[str, dict[str, Any]] = {}
        self.responses: dict[str, dict[str, Any]] = {}

    def request(self, now_s: float) -> TruthEvent:
        token = f"auth-{len(self.pending):02d}"
        payload = {"record_type": "operator_authorization_request_v1", "episode_id": self.episode_id,
                   "token_id": token, "requested_sim_time_s": now_s, "requested_utc": utc_now(),
                   "scope": "inspection_step", "validity_s": self.obligations.authorization_validity_s,
                   "purpose": "Permission to inspect the designated civilian asset",
                   "notice": "Local operator label is self-declared; this is not an identity attestation."}
        path = self.directory / f"request-{token}.json"
        _publish(path, payload)
        self.pending[token] = {"request": payload, "wall_started": time.monotonic()}
        event = TruthEvent(sim_time_s=now_s, kind="authorization_requested",
                           detail="operator approval needed",
                           payload={"token_id": token, "response_source": "operator_queue",
                                    "request_path": str(path), "request_hash": _digest(payload)})
        self._events.append(event)
        return event

    def advance_to(self, now_s: float) -> list[TruthEvent]:
        events = []
        for token, pending in self.pending.items():
            path = self.directory / f"response-{token}.json"
            if token in self.responses or not path.exists():
                continue
            row = json.loads(path.read_text())
            request = pending["request"]
            if (row.get("request_hash") != _digest(request) or row.get("episode_id") != self.episode_id
                    or row.get("token_id") != token or row.get("decision") not in {"granted", "denied"}
                    or not row.get("operator_label")):
                raise ValueError("operator response does not match the pending request")
            self.responses[token] = row
            self._requests.append({"token_id": token, "requested_at_s": request["requested_sim_time_s"],
                                   "decided_at_s": now_s, "decision": row["decision"],
                                   "expires_at_s": now_s + self.obligations.authorization_validity_s,
                                   "ordinal": float(len(self._requests))})
            self._expired_announced.add(f"granted:{token}")
            events.append(TruthEvent(
                sim_time_s=now_s,
                kind="authorization_granted" if row["decision"] == "granted" else "authorization_denied",
                detail="operator response received by runner",
                payload={"token_id": token, "response_source": "operator_queue", "decided_at_s": now_s,
                         "granted_at_s": now_s, "received_at_s": now_s,
                         "expires_at_s": now_s + self.obligations.authorization_validity_s,
                         "scope": "inspection_step", "operator_label": row["operator_label"],
                         "request_to_receipt_wall_s": time.monotonic() - pending["wall_started"],
                         "response_submitted_utc": row["submitted_utc"],
                         "note": "Wall interval includes operator, filesystem and runner polling delay."}))
        self._events.extend(events)
        events.extend(super().advance_to(now_s))
        return events

    def view(self, now_s: float) -> AuthorizationView:
        view = super().view(now_s)
        if view.status == AuthorizationStatus.ABSENT and self.pending:
            token = next(reversed(self.pending))
            return AuthorizationView(token_id=token, status=AuthorizationStatus.PENDING,
                                     requested_at_s=self.pending[token]["request"]["requested_sim_time_s"])
        if view.issuer:
            view.issuer = "local_operator_queue"
        return view

    @property
    def request_count(self) -> int:
        return len(self.pending)
