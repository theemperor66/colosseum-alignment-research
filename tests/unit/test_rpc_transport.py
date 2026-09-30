"""Transport tests for :mod:`colosseum_assurance.rpc.msgpack_rpc`.

The server here is a deliberately hostile echo server: it can hang, answer with an error, answer a
msgid that was never sent, and interleave notifications. Those are the cases that turn into silent
wrong data or an unbounded hang in a long experiment.
"""

from __future__ import annotations

import socket
import socketserver
import threading
import time
from collections.abc import Iterator

import msgpack
import pytest

from colosseum_assurance.rpc.msgpack_rpc import (
    MsgpackRpcClient,
    RpcError,
    RpcProtocolError,
    RpcTimeout,
    RpcTransportError,
    as_bytes,
)

REQUEST = 0
RESPONSE = 1
NOTIFY = 2


class _EchoHandler(socketserver.BaseRequestHandler):
    """Answers a handful of scripted methods used to probe client behaviour."""

    def handle(self) -> None:
        server: _EchoServer = self.server  # type: ignore[assignment]
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        self.request.settimeout(0.5)
        while not server.stop_event.is_set():
            try:
                chunk = self.request.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                return
            unpacker.feed(chunk)
            for message in unpacker:
                kind = message[0]
                if kind == NOTIFY:
                    server.notifications.append(message[1])
                    continue
                _, msgid, method, params = message
                server.calls.append(method)
                if method == "hang":
                    # Black hole: accepted, never answered, but the connection keeps serving. This is
                    # the shape that hangs a client forever when calls have no deadline.
                    continue
                if method == "boom":
                    self._send([RESPONSE, msgid, ["rpc error", "method refused on purpose"], None])
                    continue
                if method == "stray_then_reply":
                    # A late reply for a msgid the client already abandoned must not be handed out.
                    self._send([RESPONSE, msgid - 1, None, "stale"])
                    self._send([RESPONSE, msgid, None, "fresh"])
                    continue
                if method == "unknown_msgid":
                    self._send([RESPONSE, msgid + 5000, None, "never requested"])
                    continue
                if method == "notify_then_reply":
                    self._send([NOTIFY, "server_says", ["hello"]])
                    self._send([RESPONSE, msgid, None, "after_notification"])
                    continue
                if method == "bytes":
                    self._send([RESPONSE, msgid, None, b"\xff\xfe\x00raw"])
                    continue
                self._send([RESPONSE, msgid, None, list(params)])

    def _send(self, payload: list) -> None:
        self.request.sendall(msgpack.packb(payload, use_bin_type=True))


class _EchoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _EchoHandler)
        self.stop_event = threading.Event()
        self.calls: list[str] = []
        self.notifications: list[str] = []


@pytest.fixture()
def echo_server() -> Iterator[_EchoServer]:
    server = _EchoServer()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@pytest.fixture()
def client(echo_server: _EchoServer) -> Iterator[MsgpackRpcClient]:
    host, port = echo_server.server_address[0], echo_server.server_address[1]
    with MsgpackRpcClient(str(host), int(port), connect_timeout_s=2.0, call_timeout_s=2.0) as rpc:
        yield rpc


def test_round_trip_returns_the_same_parameters(client: MsgpackRpcClient) -> None:
    assert client.call("echo", 1, "two", [3.5, True]) == [1, "two", [3.5, True]]
    assert client.call("echo") == []


def test_msgids_increase_monotonically(client: MsgpackRpcClient) -> None:
    first = client.send_request("echo", "a")
    second = client.send_request("echo", "b")
    third = client.send_request("echo", "c")
    assert [first, second, third] == sorted({first, second, third})
    assert second == first + 1 and third == second + 1
    # Replies may be collected out of order; each one still reaches its own caller.
    assert client.receive_response(third) == ["c"]
    assert client.receive_response(first) == ["a"]
    assert client.receive_response(second) == ["b"]


def test_server_error_becomes_rpc_error_with_the_payload(client: MsgpackRpcClient) -> None:
    with pytest.raises(RpcError) as excinfo:
        client.call("boom")
    error = excinfo.value
    assert error.method == "boom"
    assert error.payload == ["rpc error", "method refused on purpose"]
    assert "method refused on purpose" in error.text


def test_timeout_is_bounded_and_the_client_stays_usable(client: MsgpackRpcClient) -> None:
    started = time.monotonic()
    with pytest.raises(RpcTimeout) as excinfo:
        client.call("hang", timeout_s=0.3)
    elapsed = time.monotonic() - started
    assert 0.25 <= elapsed < 2.0, f"timeout took {elapsed:.3f} s, outside the budget"
    assert excinfo.value.method == "hang"
    # The abandoned call must not poison the connection.
    assert client.call("echo", "still alive") == ["still alive"]


def test_late_reply_for_an_abandoned_msgid_is_discarded(client: MsgpackRpcClient) -> None:
    msgid = client.send_request("echo", "abandoned")
    client.abandon(msgid)
    assert client.call("stray_then_reply") == "fresh"
    assert client.discarded_responses >= 1


def test_response_for_a_msgid_never_sent_raises_protocol_error(client: MsgpackRpcClient) -> None:
    with pytest.raises(RpcProtocolError, match="never sent"):
        client.call("unknown_msgid", timeout_s=1.0)


def test_notifications_are_skipped_not_returned(client: MsgpackRpcClient) -> None:
    assert client.call("notify_then_reply") == "after_notification"
    assert client.received_notifications == [("server_says", ["hello"])]


def test_binary_payloads_survive_the_round_trip(client: MsgpackRpcClient) -> None:
    value = client.call("bytes")
    assert as_bytes(value) == b"\xff\xfe\x00raw"


def test_notify_reaches_the_server_without_a_reply(client: MsgpackRpcClient,
                                                   echo_server: _EchoServer) -> None:
    client.notify("ping_notification", 1)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not echo_server.notifications:
        time.sleep(0.01)
    assert echo_server.notifications == ["ping_notification"]


def test_try_receive_reports_absence_without_blocking(client: MsgpackRpcClient) -> None:
    msgid = client.send_request("hang")
    started = time.monotonic()
    done, value = client.try_receive(msgid, timeout_s=0.0)
    assert (done, value) == (False, None)
    assert time.monotonic() - started < 0.5
    client.abandon(msgid)


def test_close_is_idempotent_and_blocks_further_calls(client: MsgpackRpcClient) -> None:
    assert client.is_connected
    client.close()
    client.close()
    assert not client.is_connected
    with pytest.raises(RpcTransportError, match="not connected"):
        client.call("echo", 1)


def test_connect_to_a_closed_port_raises_transport_error() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # the port is now almost certainly free
    with pytest.raises(RpcTransportError, match="cannot connect"):
        MsgpackRpcClient("127.0.0.1", port, connect_timeout_s=1.0).connect()


def test_server_close_during_a_call_is_reported_as_a_transport_error(
    echo_server: _EchoServer,
) -> None:
    host, port = echo_server.server_address[0], echo_server.server_address[1]
    rpc = MsgpackRpcClient(str(host), int(port), connect_timeout_s=2.0, call_timeout_s=2.0).connect()
    msgid = rpc.send_request("echo", "x")
    assert rpc.receive_response(msgid) == ["x"]
    echo_server.stop_event.set()
    echo_server.shutdown()
    echo_server.server_close()
    with pytest.raises(RpcTransportError):
        for _ in range(20):
            rpc.call("echo", "y", timeout_s=0.5)
    rpc.close()
