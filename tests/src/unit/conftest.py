"""Shared helpers for the unit tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ha_mcp.client.rest_client import HomeAssistantClient


def offline_client(policy: Any = None, request: Any = None) -> HomeAssistantClient:
    """A ``HomeAssistantClient`` that never touches the network.

    ``__init__`` is skipped (no settings lookup, no real httpx client),
    ``policy`` is installed as given, and ``_request`` is replaced by
    ``request`` — by default an ``AsyncMock`` answering ``[]``, so a test
    can assert that a denied call never reached Home Assistant.
    """
    with patch.object(HomeAssistantClient, "__init__", lambda self, **kw: None):
        client = HomeAssistantClient()
    client.base_url = "http://test.local:8123"
    client.token = "test-token"
    client.timeout = 30
    client.httpx_client = MagicMock()
    client.policy = policy
    client._request = (  # type: ignore[method-assign]
        request if request is not None else AsyncMock(return_value=[])
    )
    return client
