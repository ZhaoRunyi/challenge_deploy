from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client

from . import msgpack_numpy


class WebsocketClientPolicy:
    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None) -> None:
        self.uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self.uri += f":{port}"
        self.packer = msgpack_numpy.Packer()
        self.api_key = api_key
        self.websocket, self.server_metadata = self.wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self.server_metadata

    def wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logging.info("Waiting for server at %s...", self.uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
                connection = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=None,
                )
                metadata = msgpack_numpy.unpackb(connection.recv())
                return connection, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.websocket.send(self.packer.pack(obs))
        response = self.websocket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass
