from __future__ import annotations

from typing import Any


class OpenPiPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "openpi_client is unavailable. Install the OpenPI client package into the active environment."
            ) from exc
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)

    def get_server_metadata(self) -> Any:
        return self._client.get_server_metadata()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._client.infer(payload)
