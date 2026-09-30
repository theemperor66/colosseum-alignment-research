"""Minimal msgpack-RPC transport used to speak to a Colosseum simulator.

The upstream client depends on ``msgpack-rpc-python`` (tornado based) whose Python 3.11 compatibility is
unverified, and whose default timeout is 3600 s. A guarded experiment needs short, explicit, per-call
timeouts, so this package implements the small part of the protocol we actually use.
"""

from colosseum_assurance.rpc.msgpack_rpc import (
    MsgpackRpcClient,
    MsgpackRpcError,
    RpcError,
    RpcProtocolError,
    RpcTimeout,
    RpcTransportError,
)

__all__ = [
    "MsgpackRpcClient",
    "MsgpackRpcError",
    "RpcError",
    "RpcProtocolError",
    "RpcTimeout",
    "RpcTransportError",
]
