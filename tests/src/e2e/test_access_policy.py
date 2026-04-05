"""
E2E tests for the access control policy feature.

Validates that a policy YAML file referenced via HAMCP_POLICY_FILE gates:
  1. Tool registration (tools.disabled_names / disabled_tags)
  2. REST client reads (get_states, get_entity_state)
  3. REST client writes (call_service, set_entity_state)
  4. WebSocket registry filtering (indirectly via the metadata cache)

Architecture note: the existing ``mcp_server`` fixture in conftest.py is
function-scoped and creates a fresh ``HomeAssistantSmartMCPServer`` on every
call. The server reads ``self.settings = get_global_settings()`` which is a
cached singleton. To inject per-test policy files, we set
``HAMCP_POLICY_FILE`` in the environment and reset the settings cache BEFORE
constructing the server — mirroring the URL-reset pattern already used in
conftest.py after the testcontainer starts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from fastmcp import Client

# Import test token (test_constants is on sys.path via conftest.py)
from test_constants import TEST_TOKEN

from ha_mcp.client import HomeAssistantClient
from ha_mcp.server import HomeAssistantSmartMCPServer

from .utilities.assertions import (
    assert_mcp_success,
    parse_mcp_result,
    safe_call_tool,
)
from .utilities.wait_helpers import wait_for_tool_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures — per-test MCP server with a custom policy file
# ---------------------------------------------------------------------------


def _reset_settings_cache() -> None:
    """Clear cached settings so the next Settings() pickup reads env vars."""
    import ha_mcp.config

    ha_mcp.config._settings = None


def _reset_global_policy() -> None:
    """Clear any policy installed by a previous test."""
    from ha_mcp.access_policy import set_global_policy

    set_global_policy(None)


async def _make_server_with_policy(
    base_url: str, policy_path: str | None
) -> tuple[HomeAssistantSmartMCPServer, HomeAssistantClient]:
    """Build a HomeAssistantSmartMCPServer seeing the given policy path.

    Sets HAMCP_POLICY_FILE (or clears it) in the environment, resets the
    settings + global policy caches, then constructs a fresh server. A fresh
    ``HomeAssistantClient`` is created for each server so that
    ``client.policy`` is wired correctly by ``_load_policy``.
    """
    if policy_path is not None:
        os.environ["HAMCP_POLICY_FILE"] = policy_path
    else:
        os.environ.pop("HAMCP_POLICY_FILE", None)

    _reset_settings_cache()
    _reset_global_policy()

    client = HomeAssistantClient(base_url=base_url, token=TEST_TOKEN)
    server = HomeAssistantSmartMCPServer(client=client)
    return server, client


@pytest.fixture
async def seeded_entities(ha_container_with_fresh_config):
    """Seed HA with labelled input_boolean helpers used across policy tests.

    Runs once per test (function-scoped) against the session-scoped container
    but uses a *baseline* server (no policy) so seeding itself is not gated.
    The helpers + labels are reused by subsequent tests because the container
    is session-scoped — each test re-constructs the server but the HA state
    persists.

    Returns a dict mapping a symbolic name to the created entity_id.
    """
    container_info = ha_container_with_fresh_config
    base_url = container_info["base_url"]

    # Build a policy-less server so seeding helpers is not blocked
    server, client = await _make_server_with_policy(base_url, policy_path=None)
    mcp_client = Client(server.mcp)

    # Friendly name -> expected entity_id (HA slugifies spaces to underscores).
    # We use distinctive names so slug collision is very unlikely.
    entities: dict[str, str] = {
        "alice_toy": "input_boolean.alice_toy",
        "bob_toy": "input_boolean.bob_toy",
        "shared_lamp": "input_boolean.shared_lamp",
        "parent_safe": "input_boolean.parent_safe",
    }

    async with mcp_client:
        # Check whether helpers already exist from a previous test run
        existing_ids: set[str] = set()
        for eid in entities.values():
            data = await safe_call_tool(mcp_client, "ha_get_state", {"entity_id": eid})
            if "data" in data and data.get("data") is not None:
                existing_ids.add(eid)

        # Create any missing helpers. HA slugifies the name into the entity_id,
        # so we pick names that slugify to the expected slug.
        to_create = [
            ("alice_toy", "Alice Toy"),
            ("bob_toy", "Bob Toy"),
            ("shared_lamp", "Shared Lamp"),
            ("parent_safe", "Parent Safe"),
        ]
        for slug, friendly_name in to_create:
            eid = entities[slug]
            if eid in existing_ids:
                continue
            create_result = await mcp_client.call_tool(
                "ha_config_set_helper",
                {
                    "helper_type": "input_boolean",
                    "name": friendly_name,
                },
            )
            create_data = assert_mcp_success(create_result, f"create {eid}")
            # ha_config_set_helper returns entity_id in the response.
            returned_eid = create_data.get("entity_id")
            if returned_eid and returned_eid != eid:
                # HA assigned a different slug (e.g. because of a collision).
                # Update our mapping so subsequent operations use the real id.
                entities[slug] = returned_eid
                eid = returned_eid

            # Poll until the entity is queryable in both state API and entity
            # registry (which is what the label-assignment step needs).
            await wait_for_tool_result(
                mcp_client,
                tool_name="ha_get_state",
                arguments={"entity_id": eid},
                predicate=lambda d: "data" in d and d.get("data") is not None,
                description=f"{eid} registered",
                timeout=20,
            )
            # Extra step: poll ha_get_entity until the entity_registry sees
            # it. Labels are applied via entity_registry/update which fails
            # with "Entity not found" if called too soon after helper creation.
            await wait_for_tool_result(
                mcp_client,
                tool_name="ha_get_entity",
                arguments={"entity_id": eid},
                predicate=lambda d: (
                    d.get("success") is True and d.get("entity_entry") is not None
                ),
                description=f"{eid} in entity_registry",
                timeout=20,
            )

        # Ensure labels exist. ha_config_set_label creates-by-name; if the label
        # already exists (e.g. from a previous test run against the same HA
        # container) HA raises a duplicate-name error which surfaces as a
        # ToolError. Use safe_call_tool to treat both paths uniformly and fall
        # back to lookup in either case.
        label_name_to_id: dict[str, str] = {}
        for label_name in ("owner:alice", "owner:bob", "owner:parents"):
            create_data = await safe_call_tool(
                mcp_client,
                "ha_config_set_label",
                {"name": label_name},
            )
            label_id = create_data.get("label_id") if create_data.get("success") else None
            if not label_id:
                # Already exists (or create failed) — look it up in the registry
                list_data = await safe_call_tool(
                    mcp_client, "ha_config_get_label", {}
                )
                for lbl in list_data.get("labels", []):
                    if lbl.get("name") == label_name:
                        label_id = lbl.get("label_id")
                        break
            assert label_id, f"Could not create or find label {label_name}: {create_data}"
            label_name_to_id[label_name] = label_id

        # Assign labels to entities (idempotent via label_operation="set")
        label_assignments = {
            "input_boolean.alice_toy": [label_name_to_id["owner:alice"]],
            "input_boolean.bob_toy": [label_name_to_id["owner:bob"]],
            "input_boolean.parent_safe": [label_name_to_id["owner:parents"]],
            # shared_lamp intentionally has no owner label
        }
        for eid, label_ids in label_assignments.items():
            set_result = await mcp_client.call_tool(
                "ha_set_entity",
                {
                    "entity_id": eid,
                    "labels": label_ids,
                    "label_operation": "set",
                },
            )
            assert_mcp_success(set_result, f"label {eid}")

    # Clean up server state after yielding (client will be reused if tests
    # re-create their own). Do NOT delete the helpers — container is
    # session-scoped and other policy tests depend on them.
    yield {
        "entities": entities,
        "labels": label_name_to_id,
    }

    # Clear policy cache/state for next fixture user
    _reset_global_policy()
    await client.close()


@pytest.fixture
def policy_file_factory(tmp_path: Path):
    """Factory to write a policy YAML file to tmp_path and return its path."""

    def _make(yaml_content: str, filename: str = "policy.yaml") -> str:
        policy_path = tmp_path / filename
        policy_path.write_text(yaml_content, encoding="utf-8")
        return str(policy_path)

    return _make


@pytest.fixture
async def policy_mcp_client(
    ha_container_with_fresh_config, seeded_entities, policy_file_factory
):
    """Build an MCP client wired to a per-test policy file.

    The fixture returns a *factory coroutine*: the test supplies the policy
    YAML string, gets back an async context manager yielding the MCP client.
    """
    container_info = ha_container_with_fresh_config
    base_url = container_info["base_url"]

    created_clients: list[tuple[Client, HomeAssistantClient]] = []

    class _PolicyClient:
        def __init__(self, mcp_client: Client, ha_client: HomeAssistantClient) -> None:
            self.mcp = mcp_client
            self._ha = ha_client

    async def _build(policy_yaml: str | None) -> _PolicyClient:
        if policy_yaml is None:
            policy_path = None
        else:
            policy_path = policy_file_factory(policy_yaml)
        server, ha_client = await _make_server_with_policy(base_url, policy_path)
        mcp_client = Client(server.mcp)
        await mcp_client.__aenter__()
        created_clients.append((mcp_client, ha_client))
        return _PolicyClient(mcp_client, ha_client)

    yield _build

    # Tear down
    for mcp_client, ha_client in created_clients:
        try:
            await mcp_client.__aexit__(None, None, None)
        except Exception as exc:
            logger.debug("policy_mcp_client teardown: %s", exc)
        await ha_client.close()
    _reset_global_policy()
    _reset_settings_cache()
    os.environ.pop("HAMCP_POLICY_FILE", None)


# ---------------------------------------------------------------------------
# Policy YAML builders
# ---------------------------------------------------------------------------


def _alice_policy_yaml() -> str:
    """Policy allowing Alice to read+write her own entities, and read shared_*."""
    return """
version: 1
default_action: deny
entities:
  allow:
    - labels: ["owner:alice"]
    - entity_globs: ["input_boolean.shared_*"]
  write_allow:
    - labels: ["owner:alice"]
tools:
  disabled_names: ["ha_config_set_yaml"]
"""


# ---------------------------------------------------------------------------
# Helper: extract ACCESS_DENIED code from a failed result
# ---------------------------------------------------------------------------


def _error_code(data: dict) -> str | None:
    """Pull the structured error code out of a parsed MCP failure dict."""
    err = data.get("error")
    if isinstance(err, dict):
        return err.get("code")
    return None


def _is_access_denied(data: dict) -> bool:
    """True if the result is a structured ACCESS_DENIED failure."""
    if data.get("success") is not False:
        return False
    return _error_code(data) == "ACCESS_DENIED"


def _extract_search_results(data: dict) -> list[dict]:
    """Pull the list of search-result entities from an ha_search_entities return."""
    if "data" in data and isinstance(data["data"], dict):
        return data["data"].get("results", []) or []
    return data.get("results", []) or []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAccessPolicy:
    """End-to-end tests for HAMCP_POLICY_FILE gating."""

    async def test_policy_hides_other_entities_from_search(
        self, policy_mcp_client
    ) -> None:
        """Search surfaces only entities the policy allows; others are hidden."""
        client = await policy_mcp_client(_alice_policy_yaml())

        search_result = await client.mcp.call_tool(
            "ha_search_entities",
            {
                "query": "toy",
                "domain_filter": "input_boolean",
                "limit": 20,
                "exact_match": False,
            },
        )
        data = parse_mcp_result(search_result)
        results = _extract_search_results(data)
        entity_ids = {r.get("entity_id") for r in results}

        logger.info("search returned entity_ids: %s", entity_ids)
        assert "input_boolean.alice_toy" in entity_ids, (
            f"alice_toy should be visible under alice policy, got {entity_ids}"
        )
        assert "input_boolean.bob_toy" not in entity_ids, (
            f"bob_toy must be hidden under alice policy, got {entity_ids}"
        )
        assert "input_boolean.parent_safe" not in entity_ids, (
            f"parent_safe must be hidden under alice policy, got {entity_ids}"
        )

    async def test_policy_denies_get_state_for_other_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_state on a forbidden entity returns an ACCESS_DENIED error."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_state",
            {"entity_id": "input_boolean.bob_toy"},
        )

        assert _is_access_denied(data), (
            f"ha_get_state on forbidden entity should return ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_call_service_for_other_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_call_service with forbidden entity_id is denied."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "input_boolean",
                "service": "toggle",
                "entity_id": "input_boolean.bob_toy",
            },
        )

        assert _is_access_denied(data), (
            f"ha_call_service on forbidden entity should be ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_nested_target_path(self, policy_mcp_client) -> None:
        """Policy checks the nested data.target.entity_id path, not just top-level."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "input_boolean",
                "service": "toggle",
                "data": {"target": {"entity_id": "input_boolean.bob_toy"}},
            },
        )

        assert _is_access_denied(data), (
            f"Nested target.entity_id should still be gated, got {data}"
        )

    async def test_policy_allows_owned_entity(self, policy_mcp_client) -> None:
        """Alice can successfully toggle her own (owner:alice labelled) entity."""
        client = await policy_mcp_client(_alice_policy_yaml())

        # Alice's own entity: read must succeed
        state_result = await client.mcp.call_tool(
            "ha_get_state", {"entity_id": "input_boolean.alice_toy"}
        )
        state_data = parse_mcp_result(state_result)
        assert "data" in state_data and state_data["data"] is not None, (
            f"ha_get_state for alice_toy must succeed, got {state_data}"
        )

        # Alice's own entity: write (toggle) must succeed
        toggle_result = await client.mcp.call_tool(
            "ha_call_service",
            {
                "domain": "input_boolean",
                "service": "toggle",
                "entity_id": "input_boolean.alice_toy",
                "wait": False,  # don't block on state observation
            },
        )
        assert_mcp_success(toggle_result, "toggle alice_toy")

    async def test_policy_read_only_shared(self, policy_mcp_client) -> None:
        """shared_lamp is readable (matched by allow glob) but not writable."""
        client = await policy_mcp_client(_alice_policy_yaml())

        # READ: shared_lamp is allowed by the entity_globs rule
        state_result = await client.mcp.call_tool(
            "ha_get_state", {"entity_id": "input_boolean.shared_lamp"}
        )
        state_data = parse_mcp_result(state_result)
        assert "data" in state_data and state_data["data"] is not None, (
            f"ha_get_state for shared_lamp must succeed under read allow, got {state_data}"
        )

        # WRITE: shared_lamp is NOT in write_allow (only owner:alice is)
        write_data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "input_boolean",
                "service": "toggle",
                "entity_id": "input_boolean.shared_lamp",
            },
        )
        assert _is_access_denied(write_data), (
            f"shared_lamp write should be ACCESS_DENIED, got {write_data}"
        )

    async def test_policy_disabled_tool_not_registered(
        self, policy_mcp_client
    ) -> None:
        """A tool in tools.disabled_names is not registered when policy is active."""
        client = await policy_mcp_client(_alice_policy_yaml())

        tools = await client.mcp.list_tools()
        tool_names = {t.name for t in tools}

        assert "ha_config_set_yaml" not in tool_names, (
            f"ha_config_set_yaml must be removed by policy, found in: "
            f"{sorted(n for n in tool_names if 'yaml' in n.lower())}"
        )
        # Sanity check: some other tool should still be present
        assert "ha_get_state" in tool_names, (
            "ha_get_state should still be registered under alice policy"
        )


@pytest.mark.asyncio
class TestAccessPolicyBypassFixes:
    """Tests for bypass paths closed in commits 876dacd and 56ab12d.

    Covers gates added to the client layer (logbook REST endpoint, entity-
    scoped WS commands) and the tool layer (history/statistics, camera
    snapshots, calendar reads, hard-disabled template/integration tools).
    """

    # ---- Client layer: logbook REST endpoint (commit 876dacd) ------------

    async def test_policy_denies_logbook_for_forbidden_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_logs(source='logbook', entity_id=<bob's>) is denied."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_logs",
            {
                "source": "logbook",
                "entity_id": "input_boolean.bob_toy",
                "hours_back": 1,
            },
        )

        assert _is_access_denied(data), (
            f"Logbook read for forbidden entity should be ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_logbook_without_entity_filter(
        self, policy_mcp_client
    ) -> None:
        """ha_get_logs(source='logbook') with no entity_id is denied under policy."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_logs",
            {"source": "logbook", "hours_back": 1},
        )

        assert _is_access_denied(data), (
            f"Unfiltered logbook read should be ACCESS_DENIED, got {data}"
        )
        err = data.get("error") or {}
        details = (err.get("details") or "") if isinstance(err, dict) else ""
        assert "entity_id" in details.lower(), (
            f"Denial should mention entity_id filter requirement, got details={details!r}"
        )

    async def test_policy_allows_logbook_for_owned_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_logs(source='logbook', entity_id=<alice's>) succeeds."""
        client = await policy_mcp_client(_alice_policy_yaml())

        result = await client.mcp.call_tool(
            "ha_get_logs",
            {
                "source": "logbook",
                "entity_id": "input_boolean.alice_toy",
                "hours_back": 1,
            },
        )
        data = parse_mcp_result(result)
        # Response is wrapped: {"data": {...}, "metadata": {...}} by
        # add_timezone_metadata. Inner dict has success=True.
        inner = data.get("data") if isinstance(data.get("data"), dict) else data
        assert inner.get("success") is True, (
            f"Logbook for owned entity should succeed, got {data}"
        )

    async def test_policy_filters_expose_entity_list(
        self, policy_mcp_client
    ) -> None:
        """ha_get_entity_exposure listing must not leak bob/parents entities.

        The `homeassistant/expose_entity/list` WS response is filtered by
        _apply_policy_to_ws_response so denied entity_ids are dropped from
        the exposed_entities map. A fresh HA container typically has no
        custom exposures, so we only assert that denied entities are absent
        (not that any particular entity is present).
        """
        client = await policy_mcp_client(_alice_policy_yaml())

        result = await client.mcp.call_tool("ha_get_entity_exposure", {})
        data = parse_mcp_result(result)
        assert data.get("success") is True, (
            f"expose_entity listing should succeed, got {data}"
        )
        exposed_entities = data.get("exposed_entities") or {}
        assert "input_boolean.bob_toy" not in exposed_entities, (
            f"bob_toy must be filtered out of exposed_entities, got {list(exposed_entities)}"
        )
        assert "input_boolean.parent_safe" not in exposed_entities, (
            f"parent_safe must be filtered out of exposed_entities, got {list(exposed_entities)}"
        )

    # ---- Tool layer: history & statistics (commit 56ab12d) ---------------

    async def test_policy_denies_get_history_for_forbidden_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_history for bob's entity is denied."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_history",
            {"entity_ids": "input_boolean.bob_toy"},
        )

        assert _is_access_denied(data), (
            f"ha_get_history for forbidden entity should be ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_get_history_mixed_allowed_denied(
        self, policy_mcp_client
    ) -> None:
        """ha_get_history denies the whole call when ANY entity is forbidden."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_history",
            {"entity_ids": "input_boolean.alice_toy,input_boolean.bob_toy"},
        )

        assert _is_access_denied(data), (
            f"Mixed allowed+denied history query should fail whole call, got {data}"
        )

    async def test_policy_allows_get_history_for_owned_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_history for alice's entity is not denied (HA may return empty)."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_history",
            {"entity_ids": "input_boolean.alice_toy"},
        )

        assert not _is_access_denied(data), (
            f"ha_get_history for owned entity should NOT be ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_get_statistics_for_forbidden_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_statistics for bob's entity is denied."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_statistics",
            {"entity_ids": "input_boolean.bob_toy"},
        )

        assert _is_access_denied(data), (
            f"ha_get_statistics for forbidden entity should be ACCESS_DENIED, got {data}"
        )

    # ---- Tool layer: camera & calendar (commit 56ab12d) ------------------

    async def test_policy_denies_camera_snapshot_for_forbidden_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_get_camera_image for an entity outside scope is denied.

        Uses a fabricated entity_id; under default_action=deny it will be
        denied before the camera_proxy HTTP call is even attempted.
        """
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_get_camera_image",
            {"entity_id": "camera.bob_cam"},
        )

        assert _is_access_denied(data), (
            f"ha_get_camera_image for forbidden entity should be ACCESS_DENIED, got {data}"
        )

    async def test_policy_denies_calendar_read_for_forbidden_entity(
        self, policy_mcp_client
    ) -> None:
        """ha_config_get_calendar_events for a denied calendar entity is blocked."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_config_get_calendar_events",
            {"entity_id": "calendar.bob"},
        )

        assert _is_access_denied(data), (
            f"Calendar read for forbidden entity should be ACCESS_DENIED, got {data}"
        )

    # ---- Tool layer: hard-disabled tools (commit 56ab12d) ----------------

    async def test_policy_hard_disables_eval_template(
        self, policy_mcp_client
    ) -> None:
        """ha_eval_template is refused under any active policy."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_eval_template",
            {"template": "{{ states('light.anything') }}"},
        )

        assert _is_access_denied(data), (
            f"ha_eval_template should be hard-disabled under policy, got {data}"
        )
        err = data.get("error") or {}
        details = (err.get("details") or "") if isinstance(err, dict) else ""
        assert "disabled_names" in details.lower(), (
            f"Denial should point to tools.disabled_names, got details={details!r}"
        )

    async def test_policy_hard_disables_get_integration(
        self, policy_mcp_client
    ) -> None:
        """ha_get_integration is refused under any active policy."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(client.mcp, "ha_get_integration", {})

        assert _is_access_denied(data), (
            f"ha_get_integration should be hard-disabled under policy, got {data}"
        )
        err = data.get("error") or {}
        details = (err.get("details") or "") if isinstance(err, dict) else ""
        assert "disabled_names" in details.lower(), (
            f"Denial should point to tools.disabled_names, got details={details!r}"
        )

    # ---- Targetless service-call gating (commit TBD) ---------------------

    async def test_policy_denies_targetless_service_by_default(
        self, policy_mcp_client
    ) -> None:
        """Targetless service calls are denied unless explicitly allow-listed."""
        client = await policy_mcp_client(_alice_policy_yaml())

        data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "data": {"message": "hi", "title": "test"},
            },
        )

        assert _is_access_denied(data), (
            f"Targetless service call should be ACCESS_DENIED by default, got {data}"
        )
        err = data.get("error") or {}
        details = (err.get("details") or "") if isinstance(err, dict) else ""
        assert "allow_targetless" in details.lower(), (
            f"Denial should mention services.allow_targetless, got details={details!r}"
        )

    async def test_policy_allows_exact_targetless_service(
        self, policy_mcp_client
    ) -> None:
        """A 'domain.service' entry in allow_targetless permits the exact call."""
        yaml = """
version: 1
default_action: deny
entities:
  allow:
    - labels: ["owner:alice"]
services:
  allow_targetless:
    - persistent_notification.create
"""
        client = await policy_mcp_client(yaml)

        result = await client.mcp.call_tool(
            "ha_call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "data": {
                    "message": "allowed by exact spec",
                    "title": "policy test",
                },
                "wait": False,
            },
        )
        assert_mcp_success(
            result, "targetless persistent_notification.create with exact allow"
        )

    async def test_policy_wildcard_targetless_allows_domain(
        self, policy_mcp_client
    ) -> None:
        """A 'domain.*' entry permits every service in that domain."""
        yaml = """
version: 1
default_action: deny
entities:
  allow:
    - labels: ["owner:alice"]
services:
  allow_targetless:
    - persistent_notification.*
"""
        client = await policy_mcp_client(yaml)

        result = await client.mcp.call_tool(
            "ha_call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "data": {
                    "message": "allowed by wildcard",
                    "title": "policy test",
                },
                "wait": False,
            },
        )
        assert_mcp_success(
            result, "targetless persistent_notification.create with wildcard allow"
        )

    async def test_policy_targetless_allowlist_doesnt_affect_other_services(
        self, policy_mcp_client
    ) -> None:
        """Allow-listing one service doesn't open the door to others."""
        yaml = """
version: 1
default_action: deny
entities:
  allow:
    - labels: ["owner:alice"]
services:
  allow_targetless:
    - persistent_notification.create
"""
        client = await policy_mcp_client(yaml)

        # Listed service: permitted
        data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "homeassistant",
                "service": "check_config",
            },
        )
        assert _is_access_denied(data), (
            f"homeassistant.check_config is NOT in allow_targetless → must be denied, got {data}"
        )


@pytest.mark.asyncio
class TestAccessPolicyBackwardCompat:
    """Verify server behavior is unchanged when no policy file is set."""

    async def test_no_policy_allows_all_entities(self, policy_mcp_client) -> None:
        """With HAMCP_POLICY_FILE unset, every seeded entity is accessible."""
        client = await policy_mcp_client(None)

        # All four seeded entities should be visible/readable
        for eid in (
            "input_boolean.alice_toy",
            "input_boolean.bob_toy",
            "input_boolean.shared_lamp",
            "input_boolean.parent_safe",
        ):
            state_result = await client.mcp.call_tool(
                "ha_get_state", {"entity_id": eid}
            )
            state_data = parse_mcp_result(state_result)
            assert "data" in state_data and state_data["data"] is not None, (
                f"Without a policy, {eid} must be readable, got {state_data}"
            )

        # ha_config_set_yaml should still be registered (not policy-gated)
        tools = await client.mcp.list_tools()
        tool_names = {t.name for t in tools}
        # ha_config_set_yaml registration depends on ENABLE_YAML_CONFIG_EDITING
        # flag (set by the session fixture). Both behaviors are acceptable —
        # the important thing is that a policy is NOT filtering it out.
        logger.info(
            "no-policy tool registration sanity: %d tools registered",
            len(tool_names),
        )

        # Writes to any owner's entity should be allowed
        toggle_result = await client.mcp.call_tool(
            "ha_call_service",
            {
                "domain": "input_boolean",
                "service": "toggle",
                "entity_id": "input_boolean.bob_toy",
                "wait": False,
            },
        )
        assert_mcp_success(
            toggle_result, "toggle bob_toy without policy (backward compat)"
        )

    async def test_no_policy_allows_new_gated_tools(
        self, policy_mcp_client
    ) -> None:
        """New bypass-fix gates are no-ops when no policy is active.

        Covers the tools gated in commits 876dacd and 56ab12d —
        require_can_read / require_policy_disabled must be no-ops when
        client.policy is None, so all gates pass through cleanly.
        """
        client = await policy_mcp_client(None)

        # ha_get_history: passes for any entity when no policy active
        history_data = await safe_call_tool(
            client.mcp,
            "ha_get_history",
            {"entity_ids": "input_boolean.bob_toy"},
        )
        assert not _is_access_denied(history_data), (
            f"ha_get_history must NOT be ACCESS_DENIED without a policy, got {history_data}"
        )

        # ha_eval_template: hard-disable gate is a no-op without a policy
        tmpl_data = await safe_call_tool(
            client.mcp,
            "ha_eval_template",
            {"template": "{{ 1 + 1 }}"},
        )
        assert not _is_access_denied(tmpl_data), (
            f"ha_eval_template must NOT be ACCESS_DENIED without a policy, got {tmpl_data}"
        )

        # ha_get_integration: hard-disable gate is a no-op without a policy
        integration_data = await safe_call_tool(
            client.mcp, "ha_get_integration", {}
        )
        assert not _is_access_denied(integration_data), (
            f"ha_get_integration must NOT be ACCESS_DENIED without a policy, "
            f"got {integration_data}"
        )

        # Targetless service calls: the allow_targetless gate is a no-op
        # without a policy, so persistent_notification.create passes through.
        notif_data = await safe_call_tool(
            client.mcp,
            "ha_call_service",
            {
                "domain": "persistent_notification",
                "service": "create",
                "data": {"message": "no-policy", "title": "compat"},
                "wait": False,
            },
        )
        assert not _is_access_denied(notif_data), (
            f"Targetless persistent_notification.create must NOT be denied without "
            f"a policy, got {notif_data}"
        )
