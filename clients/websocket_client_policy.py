"""OpenPI websocket policy protocol with local lifecycle controls.

The request/response sequence and MessagePack payloads follow Physical
Intelligence's ``openpi-client`` implementation:

* repository: https://github.com/Physical-Intelligence/openpi
* audited commit: ``15a9616a00943ada6c20a0f158e3adb39df2ccac``
* upstream path:
  ``packages/openpi-client/src/openpi_client/websocket_client_policy.py``

Wire behavior remains upstream-compatible: the server sends MessagePack
metadata when the socket opens, then each binary MessagePack observation gets
one binary MessagePack result.  This local adapter only adds bounded
connect/inference/close lifecycles, response validation, explicit close, and a
fresh-socket factory for retiring timed-out inference lanes.  It does not add a
payload envelope or alter the NumPy codec.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import websockets.sync.client

from . import msgpack_numpy
from .base import PolicyResponseFormatError


class WebsocketClientPolicy:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        retry_connection: bool = True,
        connect_timeout_s: float | None = None,
        inference_timeout_s: float | None = None,
        close_timeout_s: float = 0.2,
    ) -> None:
        self.uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self.uri += f":{port}"
        self.packer = msgpack_numpy.Packer()
        self.api_key = api_key
        self.retry_connection = bool(retry_connection)
        self.connect_timeout_s = connect_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self.close_timeout_s = float(close_timeout_s)
        if self.close_timeout_s <= 0.0:
            raise ValueError("close timeout must be positive")
        self.websocket, self.server_metadata = self.wait_for_server(
            retry_connection=self.retry_connection
        )

    def get_server_metadata(self) -> dict[str, Any]:
        return self.server_metadata

    def _connect_once(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        connection = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            proxy=None,
            open_timeout=self.connect_timeout_s,
            close_timeout=self.close_timeout_s,
        )
        try:
            metadata_message = connection.recv(timeout=self.inference_timeout_s)
            metadata = msgpack_numpy.unpackb(metadata_message)
        except BaseException:
            connection.close()
            raise
        return connection, metadata

    def wait_for_server(
        self,
        *,
        retry_connection: bool = True,
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logging.info("Waiting for server at %s...", self.uri)
        while True:
            try:
                return self._connect_once()
            except ConnectionRefusedError:
                if not retry_connection:
                    raise
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.websocket.send(self.packer.pack(obs))
        response = self.websocket.recv(timeout=self.inference_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        try:
            decoded = msgpack_numpy.unpackb(response)
        except Exception as exc:
            raise PolicyResponseFormatError(
                f"Inference server returned malformed MessagePack: {exc}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise PolicyResponseFormatError(
                "Inference server response must decode to an object, "
                f"got {type(decoded).__name__}"
            )
        return dict(decoded)

    def reset(self) -> None:
        pass

    def set_inference_timeout(self, timeout_s: float) -> None:
        if timeout_s <= 0.0:
            raise ValueError("inference timeout must be positive")
        self.inference_timeout_s = float(timeout_s)
        if self.connect_timeout_s is None:
            self.connect_timeout_s = float(timeout_s)

    def close(self) -> None:
        self.websocket.close()

    def new_inference_session(self) -> "WebsocketClientPolicy":
        """Open a distinct socket so an abandoned recv cannot consume a retry."""

        return type(self)(
            self.uri,
            api_key=self.api_key,
            retry_connection=False,
            connect_timeout_s=self.connect_timeout_s,
            inference_timeout_s=self.inference_timeout_s,
            close_timeout_s=self.close_timeout_s,
        )
