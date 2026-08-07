"""OpenPI-compatible NumPy MessagePack wire codec.

This module mirrors Physical Intelligence's ``openpi-client`` codec:

* repository: https://github.com/Physical-Intelligence/openpi
* audited commit: ``15a9616a00943ada6c20a0f158e3adb39df2ccac``
* upstream path: ``packages/openpi-client/src/openpi_client/msgpack_numpy.py``

The byte keys, dtype string, raw bytes, and shape representation below are wire
protocol.  Keep them compatible with upstream; local timeout and connection
lifecycle extensions belong in ``websocket_client_policy`` and must not change
this encoding.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
