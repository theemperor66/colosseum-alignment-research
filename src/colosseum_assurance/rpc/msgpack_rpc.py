"""msgpack-RPC v1 client over TCP, written directly on the ``msgpack`` package.

WHY this exists instead of ``msgpack-rpc-python`` (the package the upstream client imports):

* Upstream ``PythonClient/airsim/client.py`` builds its session with ``timeout=3600`` seconds
  (verified at commit 84fc0c1c75bc73a0135ee80a325d470577c66c52, ``client.py:17``). A guarded
  experiment must bound every simulator call, and an exceeded bound must be recorded as *incomplete*,
  never as a pass. Explicit per-call deadlines are therefore a requirement, not a nicety.
* ``msgpack-rpc-python`` pulls in tornado and passes ``pack_encoding``/``unpack_encoding`` arguments
  that current ``msgpack`` releases removed. Its behaviour on Python 3.11+ is unverified.
* We need to *send a request without waiting for its reply* (the simulator executes movement commands
  while the clock is advanced by a separate call). That is a normal msgpack-RPC pattern but it is not
  exposed cleanly by the upstream wrapper.

Wire protocol (msgpack-RPC v1, https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md):

* request      ``[0, msgid, method, params]``
* response     ``[1, msgid, error, result]``
* notification ``[2, method, params]``

Encoding notes:

* ``use_bin_type=True`` when packing, so Python ``str`` becomes msgpack ``str`` and ``bytes`` becomes
  msgpack ``bin``. That matches what the C++ server (rpclib/msgpack-c) expects.
* ``raw=False`` when unpacking, with ``unicode_errors="surrogateescape"``. Surrogate escaping is a
  lossless round trip for arbitrary bytes, so a server that sends image bytes inside a msgpack ``str``
  frame cannot crash the decoder; :func:`as_bytes` recovers the exact payload.
"""

from __future__ import annotations

import socket
import time
from types import TracebackType
from typing import Any

import msgpack

REQUEST = 0
RESPONSE = 1
NOTIFY = 2

DEFAULT_RECV_BYTES = 262144
DEFAULT_MAX_BUFFER_BYTES = 128 * 1024 * 1024


class MsgpackRpcError(RuntimeError):
    """Base class for every failure of this transport."""


class RpcTransportError(MsgpackRpcError):
    """The TCP connection failed, was refused, or was closed by the peer."""


class RpcTimeout(RpcTransportError):
    """A call did not produce a response inside its deadline."""

    def __init__(self, message: str, *, method: str = "", msgid: int | None = None,
                 elapsed_s: float | None = None) -> None:
        super().__init__(message)
        self.method = method
        self.msgid = msgid
        self.elapsed_s = elapsed_s


class RpcProtocolError(MsgpackRpcError):
    """The peer sent a frame this client cannot interpret (bad shape, unknown msgid)."""


class RpcError(MsgpackRpcError):
    """The server answered with a non-nil error field.

    ``payload`` keeps the server's error object verbatim so diagnostics can show it unmodified. A
    genuine Colosseum answers an unknown method with such an error, which is how the fixture-fake probe
    in :mod:`colosseum_assurance.sim.identity` tells a real server from our test fixture.
    """

    def __init__(self, method: str, payload: Any, *, msgid: int | None = None) -> None:
        super().__init__(f"RPC call {method!r} failed: {payload!r}")
        self.method = method
        self.payload = payload
        self.msgid = msgid

    @property
    def text(self) -> str:
        """Best-effort human text of the server error payload."""
        payload = self.payload
        if isinstance(payload, bytes):
            return payload.decode("utf-8", "replace")
        if isinstance(payload, (list, tuple)):
            return " | ".join(str(part) for part in payload)
        return str(payload)


def as_bytes(value: Any) -> bytes:
    """Recover raw bytes from a decoded msgpack value (``bin`` or surrogate-escaped ``str``)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    raise TypeError(f"cannot read {type(value).__name__} as bytes")


class MsgpackRpcClient:
    """A synchronous msgpack-RPC client with an explicit deadline on every call.

    The client is *not* thread safe: one connection belongs to one caller. The adapter owns exactly one
    client, and the runner drives the adapter from a single thread.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 10.0,
        call_timeout_s: float = 20.0,
        recv_bytes: int = DEFAULT_RECV_BYTES,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        if connect_timeout_s <= 0.0 or call_timeout_s <= 0.0:
            raise ValueError("connect_timeout_s and call_timeout_s must be positive")
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self._recv_bytes = int(recv_bytes)
        self._max_buffer_bytes = int(max_buffer_bytes)
        self._sock: socket.socket | None = None
        self._unpacker: msgpack.Unpacker | None = None
        self._next_msgid = 0
        self._inflight: dict[int, str] = {}
        self._ready: dict[int, tuple[Any, Any]] = {}
        self.discarded_responses = 0
        self.received_notifications: list[tuple[str, Any]] = []

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> MsgpackRpcClient:
        """Open the TCP connection. Raises :class:`RpcTransportError` on refusal or DNS failure."""
        if self._sock is not None:
            return self
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        except OSError as exc:
            raise RpcTransportError(
                f"cannot connect to {self.host}:{self.port} within {self.connect_timeout_s:g} s: {exc}"
            ) from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._unpacker = msgpack.Unpacker(
            raw=False,
            strict_map_key=False,
            unicode_errors="surrogateescape",
            max_buffer_size=self._max_buffer_bytes,
        )
        return self

    def close(self) -> None:
        """Close the socket. Safe to call twice; never raises."""
        sock, self._sock = self._sock, None
        self._unpacker = None
        self._inflight.clear()
        self._ready.clear()
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def __enter__(self) -> MsgpackRpcClient:
        return self.connect()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    # ------------------------------------------------------------------ sending
    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise RpcTransportError("client is not connected; call connect() first")
        return self._sock

    def send_request(self, method: str, *params: Any) -> int:
        """Send a request and return its msgid without waiting for the response.

        Used for simulator commands that the server executes while the caller advances the simulation
        clock through a separate call. The reply is collected later with :meth:`receive_response` or
        discarded with :meth:`abandon`.
        """
        sock = self._require_socket()
        msgid = self._next_msgid
        self._next_msgid += 1
        payload = msgpack.packb([REQUEST, msgid, method, list(params)], use_bin_type=True)
        try:
            sock.settimeout(self.call_timeout_s)
            sock.sendall(payload)
        except OSError as exc:
            raise RpcTransportError(f"sending {method!r} failed: {exc}") from exc
        self._inflight[msgid] = method
        return msgid

    def notify(self, method: str, *params: Any) -> None:
        """Send a notification. The protocol defines no reply for it."""
        sock = self._require_socket()
        payload = msgpack.packb([NOTIFY, method, list(params)], use_bin_type=True)
        try:
            sock.settimeout(self.call_timeout_s)
            sock.sendall(payload)
        except OSError as exc:
            raise RpcTransportError(f"sending notification {method!r} failed: {exc}") from exc

    def abandon(self, msgid: int) -> None:
        """Stop waiting for ``msgid``; a late response for it is discarded instead of confusing a call."""
        self._inflight.pop(msgid, None)
        self._ready.pop(msgid, None)

    # ------------------------------------------------------------------ receiving
    def call(self, method: str, *params: Any, timeout_s: float | None = None) -> Any:
        """Send ``method`` and wait for its result inside ``timeout_s`` (default: the client timeout)."""
        msgid = self.send_request(method, *params)
        return self.receive_response(msgid, timeout_s=timeout_s)

    def receive_response(self, msgid: int, *, timeout_s: float | None = None) -> Any:
        """Wait for the response to ``msgid``.

        On timeout the msgid is abandoned, so the connection stays usable: a late reply is recognised
        and dropped instead of being handed to the next caller.
        """
        budget = self.call_timeout_s if timeout_s is None else float(timeout_s)
        method = self._inflight.get(msgid, "<unknown>")
        started = time.monotonic()
        deadline = started + budget
        ready = self._ready.pop(msgid, None)
        if ready is not None:
            return self._unwrap(method, msgid, ready)
        while True:
            try:
                frame = self._read_frame(deadline)
            except RpcTimeout:
                self.abandon(msgid)
                raise RpcTimeout(
                    f"no response to {method!r} (msgid {msgid}) within {budget:g} s",
                    method=method, msgid=msgid, elapsed_s=time.monotonic() - started,
                ) from None
            got_id, error, result = frame
            if got_id == msgid:
                self._inflight.pop(msgid, None)
                return self._unwrap(method, msgid, (error, result))
            self._route_other(got_id, error, result)

    def try_receive(self, msgid: int, *, timeout_s: float = 0.0) -> tuple[bool, Any]:
        """Poll for the response to ``msgid``. Returns ``(False, None)`` when it has not arrived yet."""
        method = self._inflight.get(msgid, "<unknown>")
        ready = self._ready.pop(msgid, None)
        if ready is not None:
            return True, self._unwrap(method, msgid, ready)
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while True:
            try:
                frame = self._read_frame(deadline)
            except RpcTimeout:
                return False, None
            got_id, error, result = frame
            if got_id == msgid:
                self._inflight.pop(msgid, None)
                return True, self._unwrap(method, msgid, (error, result))
            self._route_other(got_id, error, result)

    # ------------------------------------------------------------------ internals
    def _unwrap(self, method: str, msgid: int, frame: tuple[Any, Any]) -> Any:
        error, result = frame
        if error is not None:
            raise RpcError(method, error, msgid=msgid)
        return result

    def _route_other(self, got_id: int, error: Any, result: Any) -> None:
        """Handle a response whose msgid is not the one being awaited.

        Msgids only grow, so the classification needs no per-call bookkeeping: an id still in flight
        belongs to another outstanding call, an id below the next msgid is stale (abandoned after a
        timeout, duplicated, or already delivered), and an id at or above the next msgid was never sent
        by this client and means the peer is not following the protocol.
        """
        if got_id in self._inflight:
            self._ready[got_id] = (error, result)
            return
        if 0 <= got_id < self._next_msgid:
            self.discarded_responses += 1
            return
        raise RpcProtocolError(
            f"response for msgid {got_id} which this client never sent "
            f"(next msgid would be {self._next_msgid}); the peer is not following msgpack-RPC v1"
        )

    def _read_frame(self, deadline: float) -> tuple[int, Any, Any]:
        """Read the next response frame, skipping notifications. Raises :class:`RpcTimeout`."""
        while True:
            message = self._next_message(deadline)
            if not isinstance(message, (list, tuple)) or len(message) < 3:
                raise RpcProtocolError(f"malformed msgpack-RPC frame: {message!r}")
            kind = message[0]
            if kind == NOTIFY:
                self.received_notifications.append((str(message[1]), message[2]))
                continue
            if kind != RESPONSE or len(message) != 4:
                raise RpcProtocolError(
                    f"expected a msgpack-RPC response [1, msgid, error, result], got {message!r}"
                )
            return int(message[1]), message[2], message[3]

    def _next_message(self, deadline: float) -> Any:
        sock = self._require_socket()
        unpacker = self._unpacker
        if unpacker is None:  # pragma: no cover - set together with _sock in connect()
            raise RpcTransportError("client is not connected; call connect() first")
        while True:
            try:
                return next(unpacker)
            except StopIteration:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RpcTimeout("deadline exceeded while waiting for a response")
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(self._recv_bytes)
            except TimeoutError as exc:  # socket.timeout is an alias of TimeoutError on 3.10+
                raise RpcTimeout("deadline exceeded while waiting for a response") from exc
            except OSError as exc:
                raise RpcTransportError(f"reading from {self.host}:{self.port} failed: {exc}") from exc
            if not chunk:
                raise RpcTransportError(
                    f"{self.host}:{self.port} closed the connection while a response was outstanding"
                )
            unpacker.feed(chunk)
