"""Unit tests for the fine-grained access control policy module.

Covers:
- Policy schema loading (YAML + Pydantic validation)
- Entity rule matching primitives (area, label, glob, device_id)
- Precedence rules (deny > allow > default_action)
- Write-rule semantics (write_allow inheritance, write_deny, readonly_mode)
- Tool gating (disabled_names, disabled_tags, readonly destructive shortcut)
- Entity + device label/area union
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from ha_mcp.access_policy import (
    AccessPolicy,
    EntitiesPolicy,
    EntityMeta,
    EntityMetadataCache,
    EntityRule,
    PolicyConfig,
    PolicyLoadError,
    ServicesPolicy,
    ToolsPolicy,
    get_global_policy,
    load_policy_from_file,
    set_global_policy,
)


class FakeCache(EntityMetadataCache):
    """Test double that bypasses websocket fetches.

    - Skips EntityMetadataCache.__init__ (no client needed).
    - Presets _fetched_at to +inf so ensure_fresh() and the base TTL check
      always short-circuit ("never expire"), letting each test control the
      cache contents directly.
    - If EntityMetadataCache gains new required instance state, update this
      constructor — tests will surface the breakage.
    """

    def __init__(self, entities: dict[str, EntityMeta] | None = None):
        import asyncio

        self._entities = entities or {}
        self._fetched_at = float("inf")
        self._lock = asyncio.Lock()

    async def ensure_fresh(self) -> None:  # override — no-op
        return

    async def _refresh(self) -> None:  # pragma: no cover — never called
        return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def meta(
    entity_id: str,
    area_id: str | None = None,
    labels: set[str] | None = None,
    device_id: str | None = None,
) -> EntityMeta:
    return EntityMeta(
        entity_id=entity_id,
        area_id=area_id,
        labels=labels or set(),
        device_id=device_id,
    )


def make_policy(
    cfg: PolicyConfig, entities: dict[str, EntityMeta] | None = None
) -> AccessPolicy:
    return AccessPolicy(cfg, FakeCache(entities))


# ---------------------------------------------------------------------------
# Rule matching primitives
# ---------------------------------------------------------------------------


class TestRuleMatching:
    async def test_glob_match(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(entity_globs=["light.alice_*"])]
            )
        )
        entities = {
            "light.alice_bed": meta("light.alice_bed"),
            "light.bob_bed": meta("light.bob_bed"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.alice_bed")
        assert not await p.can_read("light.bob_bed")

    async def test_area_match(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(areas=["alice_bedroom"])])
        )
        entities = {
            "light.x": meta("light.x", area_id="alice_bedroom"),
            "light.y": meta("light.y", area_id="bob_bedroom"),
            "light.z": meta("light.z", area_id=None),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.x")
        assert not await p.can_read("light.y")
        assert not await p.can_read("light.z")

    async def test_label_match_intersection(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(labels=["owner:alice"])])
        )
        entities = {
            "light.a": meta("light.a", labels={"owner:alice"}),
            "light.b": meta("light.b", labels={"owner:bob"}),
            "light.c": meta("light.c", labels={"owner:alice", "kid-safe"}),
            "light.d": meta("light.d", labels=set()),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.a")
        assert not await p.can_read("light.b")
        assert await p.can_read("light.c")
        assert not await p.can_read("light.d")

    async def test_device_id_match(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(device_ids=["dev-abc"])])
        )
        entities = {
            "media_player.sonos_alice": meta(
                "media_player.sonos_alice", device_id="dev-abc"
            ),
            "media_player.sonos_bob": meta(
                "media_player.sonos_bob", device_id="dev-xyz"
            ),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("media_player.sonos_alice")
        assert not await p.can_read("media_player.sonos_bob")

    async def test_empty_rule_matches_nothing(self):
        """A rule with all-empty primitives should never match."""
        cfg = PolicyConfig(entities=EntitiesPolicy(allow=[EntityRule()]))
        entities = {
            "light.x": meta(
                "light.x", area_id="kitchen", labels={"owner:alice"}, device_id="d1"
            )
        }
        p = make_policy(cfg, entities)
        assert not await p.can_read("light.x")

    async def test_multiple_allow_rules_or_across_list(self):
        """Two separate rules in `allow` are OR'd — matching either permits read."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[
                    EntityRule(areas=["kitchen"]),
                    EntityRule(labels=["owner:alice"]),
                ]
            )
        )
        entities = {
            "light.k": meta("light.k", area_id="kitchen"),
            "light.a": meta("light.a", labels={"owner:alice"}),
            "light.both": meta(
                "light.both", area_id="kitchen", labels={"owner:alice"}
            ),
            "light.neither": meta("light.neither", area_id="garage"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.k")
        assert await p.can_read("light.a")
        assert await p.can_read("light.both")
        assert not await p.can_read("light.neither")

    async def test_rule_is_or_across_primitives(self):
        """A single rule matches if ANY primitive matches."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[
                    EntityRule(
                        areas=["kitchen"],
                        labels=["owner:alice"],
                        entity_globs=["weather.*"],
                    )
                ]
            )
        )
        entities = {
            "light.k": meta("light.k", area_id="kitchen"),
            "light.a": meta("light.a", labels={"owner:alice"}),
            "weather.home": meta("weather.home"),
            "light.other": meta("light.other", area_id="garage"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.k")
        assert await p.can_read("light.a")
        assert await p.can_read("weather.home")
        assert not await p.can_read("light.other")


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    async def test_deny_beats_allow(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(entity_globs=["light.*"])],
                deny=[EntityRule(entity_globs=["light.bedroom_*"])],
            )
        )
        entities = {
            "light.kitchen": meta("light.kitchen"),
            "light.bedroom_main": meta("light.bedroom_main"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.kitchen")
        assert not await p.can_read("light.bedroom_main")

    async def test_allow_beats_default_deny(self):
        cfg = PolicyConfig(
            default_action="deny",
            entities=EntitiesPolicy(allow=[EntityRule(entity_globs=["light.a"])]),
        )
        entities = {
            "light.a": meta("light.a"),
            "light.b": meta("light.b"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.a")
        assert not await p.can_read("light.b")

    async def test_default_allow_permits_unmatched(self):
        cfg = PolicyConfig(
            default_action="allow",
            entities=EntitiesPolicy(deny=[EntityRule(entity_globs=["lock.*"])]),
        )
        entities = {
            "light.anything": meta("light.anything"),
            "lock.door": meta("lock.door"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.anything")
        assert not await p.can_read("lock.door")

    async def test_default_action_defaults_to_deny(self):
        """Principle of least privilege: no default_action in YAML == deny."""
        cfg = PolicyConfig()
        assert cfg.default_action == "deny"
        entities = {"light.x": meta("light.x")}
        p = make_policy(cfg, entities)
        assert not await p.can_read("light.x")

    async def test_empty_policy_with_allow_denies_everything(self):
        """default=deny + empty allow list = nothing readable."""
        cfg = PolicyConfig()
        entities = {"light.x": meta("light.x")}
        p = make_policy(cfg, entities)
        assert not await p.can_read("light.x")


# ---------------------------------------------------------------------------
# Write rules
# ---------------------------------------------------------------------------


class TestWriteRules:
    async def test_write_allow_unset_inherits_allow(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(entity_globs=["light.*"])]),
        )
        entities = {"light.a": meta("light.a"), "switch.a": meta("switch.a")}
        p = make_policy(cfg, entities)
        assert await p.can_write("light.a")
        assert not await p.can_write("switch.a")  # not in allow → not readable

    async def test_write_allow_narrower_than_read(self):
        """Alice can see her whole bedroom but only write to her own stuff."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(areas=["alice_bedroom"])],
                write_allow=[EntityRule(labels=["owner:alice"])],
            ),
        )
        entities = {
            "light.shared_ceiling": meta(
                "light.shared_ceiling", area_id="alice_bedroom"
            ),
            "light.alice_lamp": meta(
                "light.alice_lamp",
                area_id="alice_bedroom",
                labels={"owner:alice"},
            ),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.shared_ceiling")
        assert not await p.can_write("light.shared_ceiling")
        assert await p.can_read("light.alice_lamp")
        assert await p.can_write("light.alice_lamp")

    async def test_write_deny_blocks_even_when_readable(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(entity_globs=["*"])],
                write_deny=[EntityRule(entity_globs=["climate.whole_house_*"])],
            )
        )
        entities = {
            "climate.whole_house_main": meta("climate.whole_house_main"),
            "light.x": meta("light.x"),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("climate.whole_house_main")
        assert not await p.can_write("climate.whole_house_main")
        assert await p.can_write("light.x")

    async def test_write_allow_empty_blocks_all_writes(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(entity_globs=["*"])],
                write_allow=[],
            )
        )
        entities = {"light.x": meta("light.x")}
        p = make_policy(cfg, entities)
        assert await p.can_read("light.x")
        assert not await p.can_write("light.x")

    async def test_readonly_mode_blocks_all_writes(self):
        cfg = PolicyConfig(
            readonly_mode=True,
            entities=EntitiesPolicy(allow=[EntityRule(entity_globs=["*"])]),
        )
        entities = {"light.x": meta("light.x")}
        p = make_policy(cfg, entities)
        assert await p.can_read("light.x")
        assert not await p.can_write("light.x")


# ---------------------------------------------------------------------------
# Tool gating
# ---------------------------------------------------------------------------


class TestToolGating:
    def test_disabled_by_name(self):
        cfg = PolicyConfig(tools=ToolsPolicy(disabled_names=["ha_backup_create"]))
        p = make_policy(cfg)
        assert not p.is_tool_allowed("ha_backup_create")
        assert p.is_tool_allowed("ha_get_state")

    def test_disabled_by_tag(self):
        cfg = PolicyConfig(tools=ToolsPolicy(disabled_tags=["Add-ons", "HACS"]))
        p = make_policy(cfg)
        assert not p.is_tool_allowed("ha_install_addon", tags={"Add-ons"})
        assert not p.is_tool_allowed("ha_install_hacs", tags={"HACS"})
        assert p.is_tool_allowed("ha_get_state", tags={"Entities"})
        assert p.is_tool_allowed("untagged_tool")
        assert p.is_tool_allowed("untagged_tool", tags=set())

    def test_readonly_mode_blocks_any_write_tool(self):
        """Only tools explicitly marked readOnlyHint=True are allowed under readonly_mode."""
        cfg = PolicyConfig(readonly_mode=True)
        p = make_policy(cfg)
        # Explicitly read-only tool: allowed
        assert p.is_tool_allowed(
            "ha_get_state", annotations={"readOnlyHint": True}
        )
        # No annotations / no readOnlyHint → treated as a write → blocked
        assert not p.is_tool_allowed("ha_call_service", annotations={})
        assert not p.is_tool_allowed("ha_call_service", annotations=None)
        # Non-destructive write (e.g. creating a helper) is still a write → blocked
        assert not p.is_tool_allowed(
            "ha_config_set_helper", annotations={"destructiveHint": False}
        )

    def test_readonly_mode_off_allows_writes(self):
        cfg = PolicyConfig(readonly_mode=False)
        p = make_policy(cfg)
        assert p.is_tool_allowed("ha_call_service", annotations={})


# ---------------------------------------------------------------------------
# Service-call targetless allow-list
# ---------------------------------------------------------------------------


class TestTargetlessServices:
    """services.allow_targetless controls whether targetless calls pass."""

    def test_empty_list_denies_all(self):
        p = make_policy(PolicyConfig())
        assert not p.allows_targetless_service("homeassistant", "restart")
        assert not p.allows_targetless_service("notify", "telegram")

    def test_exact_domain_service_match(self):
        cfg = PolicyConfig(
            services=ServicesPolicy(
                allow_targetless=["persistent_notification.create"]
            )
        )
        p = make_policy(cfg)
        assert p.allows_targetless_service("persistent_notification", "create")
        # Different service in same domain → denied
        assert not p.allows_targetless_service("persistent_notification", "dismiss")
        # Different domain → denied
        assert not p.allows_targetless_service("notify", "create")

    def test_wildcard_covers_whole_domain(self):
        cfg = PolicyConfig(
            services=ServicesPolicy(allow_targetless=["notify.*"])
        )
        p = make_policy(cfg)
        assert p.allows_targetless_service("notify", "telegram")
        assert p.allows_targetless_service("notify", "persistent_notification")
        # Other domain unaffected
        assert not p.allows_targetless_service("homeassistant", "restart")

    def test_mixed_exact_and_wildcard(self):
        cfg = PolicyConfig(
            services=ServicesPolicy(
                allow_targetless=["notify.*", "persistent_notification.create"]
            )
        )
        p = make_policy(cfg)
        assert p.allows_targetless_service("notify", "anything")
        assert p.allows_targetless_service("persistent_notification", "create")
        assert not p.allows_targetless_service("persistent_notification", "dismiss")

    def test_readonly_mode_denies_even_listed(self):
        cfg = PolicyConfig(
            readonly_mode=True,
            services=ServicesPolicy(allow_targetless=["notify.*"]),
        )
        p = make_policy(cfg)
        # readonly_mode overrides any allow_targetless entry
        assert not p.allows_targetless_service("notify", "telegram")

    def test_invalid_spec_rejected(self):
        """Entries not matching 'domain.service' or 'domain.*' fail validation."""
        import pytest
        from pydantic import ValidationError

        for bad in ["notify", "notify.*.extra", "NOTIFY.telegram", "notify.", ".create"]:
            with pytest.raises(ValidationError):
                ServicesPolicy(allow_targetless=[bad])

    def test_domain_wildcard_shape(self):
        """'.*' is only valid as a whole-service wildcard, not a prefix."""
        import pytest
        from pydantic import ValidationError

        # Valid
        ServicesPolicy(allow_targetless=["notify.*"])
        # Invalid (no partial wildcards)
        with pytest.raises(ValidationError):
            ServicesPolicy(allow_targetless=["notify.tel*"])


# ---------------------------------------------------------------------------
# Entity + device label/area union
# ---------------------------------------------------------------------------


class TestMetadataUnion:
    """EntityMeta is pre-resolved by the cache, but test the matching honours
    the resolved state correctly (entity inherits device's area/labels)."""

    async def test_area_fallback_via_device(self):
        """Entity with no area_id still matches if its device has that area."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(areas=["kitchen"])])
        )
        # Simulate EntityMetadataCache's resolution: entity.area_id is None but
        # fell back to device.area_id="kitchen" during refresh, so the cached
        # EntityMeta.area_id is "kitchen".
        entities = {
            "sensor.fridge_temp": meta(
                "sensor.fridge_temp", area_id="kitchen", device_id="fridge-dev"
            ),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("sensor.fridge_temp")

    async def test_labels_union_entity_and_device(self):
        """Entity with device-sourced labels matches rules on those labels."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(labels=["owner:alice"])])
        )
        # Cache has already merged entity.labels ∪ device.labels
        entities = {
            "media_player.alice_sonos": meta(
                "media_player.alice_sonos",
                labels={"kid-safe", "owner:alice"},  # owner:alice came from device
                device_id="sonos-alice",
            )
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("media_player.alice_sonos")

    async def test_labels_match_by_id_or_name(self):
        """The cache expands meta.labels to include both IDs and names so a
        rule can reference a label by either spelling (HA stores IDs on
        entities but users naturally write names in YAML)."""
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(labels=["owner:alice"])]  # the human name
            )
        )
        # What the cache would actually produce after _expand_labels:
        # entity has label_id "owner_alice" → expanded to {"owner_alice", "owner:alice"}
        entities = {
            "light.alice_lamp": meta(
                "light.alice_lamp",
                labels={"owner_alice", "owner:alice"},
            ),
        }
        p = make_policy(cfg, entities)
        assert await p.can_read("light.alice_lamp")

        # And the inverse: a policy that references the ID should also work.
        cfg_by_id = PolicyConfig(
            entities=EntitiesPolicy(allow=[EntityRule(labels=["owner_alice"])])
        )
        p_by_id = make_policy(cfg_by_id, entities)
        assert await p_by_id.can_read("light.alice_lamp")


# ---------------------------------------------------------------------------
# filter_entities
# ---------------------------------------------------------------------------


class TestFilterEntities:
    async def test_filter_keeps_only_readable(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(labels=["owner:alice"])]
            )
        )
        entities = {
            "light.a1": meta("light.a1", labels={"owner:alice"}),
            "light.b1": meta("light.b1", labels={"owner:bob"}),
            "light.a2": meta("light.a2", labels={"owner:alice"}),
        }
        p = make_policy(cfg, entities)
        filtered = await p.filter_entities(
            ["light.a1", "light.b1", "light.a2", "light.missing"]
        )
        assert set(filtered) == {"light.a1", "light.a2"}

    async def test_can_read_batch_returns_decisions_for_each(self):
        cfg = PolicyConfig(
            entities=EntitiesPolicy(
                allow=[EntityRule(labels=["owner:alice"])]
            )
        )
        entities = {
            "light.a1": meta("light.a1", labels={"owner:alice"}),
            "light.b1": meta("light.b1", labels={"owner:bob"}),
        }
        p = make_policy(cfg, entities)
        result = await p.can_read_batch(["light.a1", "light.b1", "light.missing"])
        assert result == {
            "light.a1": True,
            "light.b1": False,
            "light.missing": False,  # unknown entity denied under default=deny
        }


# ---------------------------------------------------------------------------
# expand_targets (device/area/label → entity_id set)
# ---------------------------------------------------------------------------


class TestExpandTargets:
    def _policy_with_entities(self) -> AccessPolicy:
        entities = {
            "light.alice_bed": meta(
                "light.alice_bed",
                area_id="alice_bedroom",
                labels={"owner:alice"},
                device_id="dev-alice-hub",
            ),
            "sensor.alice_temp": meta(
                "sensor.alice_temp",
                area_id="alice_bedroom",
                device_id="dev-alice-hub",
            ),
            "light.bob_bed": meta(
                "light.bob_bed",
                area_id="bob_bedroom",
                labels={"owner:bob"},
                device_id="dev-bob-hub",
            ),
            "light.kitchen": meta("light.kitchen", area_id="kitchen"),
        }
        return make_policy(PolicyConfig(), entities)

    async def test_expand_by_device_id(self):
        p = self._policy_with_entities()
        result = await p.expand_targets(device_ids=["dev-alice-hub"])
        assert result == {"light.alice_bed", "sensor.alice_temp"}

    async def test_expand_by_area_id(self):
        p = self._policy_with_entities()
        result = await p.expand_targets(area_ids=["bob_bedroom"])
        assert result == {"light.bob_bed"}

    async def test_expand_by_label(self):
        p = self._policy_with_entities()
        result = await p.expand_targets(label_ids=["owner:alice"])
        assert result == {"light.alice_bed"}

    async def test_expand_union_of_targets(self):
        p = self._policy_with_entities()
        result = await p.expand_targets(
            area_ids=["kitchen"], label_ids=["owner:bob"]
        )
        assert result == {"light.kitchen", "light.bob_bed"}

    async def test_expand_empty_arguments_returns_empty_set(self):
        p = self._policy_with_entities()
        assert await p.expand_targets() == set()
        assert await p.expand_targets(device_ids=[], area_ids=[]) == set()

    async def test_expand_unknown_target_returns_empty_set(self):
        p = self._policy_with_entities()
        result = await p.expand_targets(device_ids=["ghost-device"])
        assert result == set()


# ---------------------------------------------------------------------------
# Version validation
# ---------------------------------------------------------------------------


class TestVersionValidation:
    def test_version_1_accepted(self):
        cfg = PolicyConfig(version=1)
        assert cfg.version == 1

    def test_future_version_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            PolicyConfig(version=2)

    def test_future_version_rejected_via_yaml(self, tmp_path: Path):
        policy_file = tmp_path / "future.yaml"
        policy_file.write_text("version: 99\n")
        with pytest.raises(PolicyLoadError, match="validation"):
            load_policy_from_file(policy_file)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalSingleton:
    def test_singleton_starts_unset(self):
        set_global_policy(None)  # ensure clean slate
        assert get_global_policy() is None

    def test_set_and_get_roundtrip(self):
        cfg = PolicyConfig()
        policy = AccessPolicy(cfg, FakeCache())
        try:
            set_global_policy(policy)
            assert get_global_policy() is policy
        finally:
            set_global_policy(None)

    def test_singleton_cleared_by_none(self):
        policy = AccessPolicy(PolicyConfig(), FakeCache())
        set_global_policy(policy)
        set_global_policy(None)
        assert get_global_policy() is None


# ---------------------------------------------------------------------------
# Cache behavior (TTL + invalidate + failure handling)
# ---------------------------------------------------------------------------


class RecordingClient:
    """Minimal client that returns pre-programmed registry responses."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.call_count = 0

    async def send_websocket_message(self, msg: dict[str, Any]) -> Any:
        self.call_count += 1
        # Pop two at a time (entity then device) per _refresh call
        return self._responses.pop(0)


class TestCacheTTL:
    # Each _refresh() issues 3 WS calls (entity_registry, device_registry,
    # label_registry). The empty-registry sentinel used across these tests:
    _EMPTY: ClassVar[dict[str, Any]] = {"success": True, "result": []}

    async def test_invalidate_forces_refresh(self):
        """invalidate() sets _fetched_at back to 0, so next ensure_fresh refetches."""
        # Two entities on first fetch, one on second fetch
        client = RecordingClient(
            responses=[
                {"success": True, "result": [
                    {"entity_id": "light.a", "area_id": None, "labels": [], "device_id": None},
                    {"entity_id": "light.b", "area_id": None, "labels": [], "device_id": None},
                ]},
                self._EMPTY,  # device registry
                self._EMPTY,  # label registry
                {"success": True, "result": [
                    {"entity_id": "light.a", "area_id": None, "labels": [], "device_id": None},
                ]},
                self._EMPTY,
                self._EMPTY,
            ]
        )
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        assert set(cache._entities.keys()) == {"light.a", "light.b"}
        assert client.call_count == 3

        cache.invalidate()
        await cache.ensure_fresh()
        assert set(cache._entities.keys()) == {"light.a"}
        assert client.call_count == 6

    async def test_failed_refresh_keeps_previous_cache_but_marks_fetched(self):
        """When entity_registry fetch fails after prior success, keep old cache
        but DO advance _fetched_at so we don't stampede the backend on every call."""
        client = RecordingClient(
            responses=[
                {"success": True, "result": [
                    {"entity_id": "light.a", "area_id": None, "labels": [], "device_id": None},
                ]},
                self._EMPTY,  # device
                self._EMPTY,  # label
                # Second refresh: entity_registry fails
                RuntimeError("entity registry down"),
                self._EMPTY,
                self._EMPTY,
            ]
        )
        cache = EntityMetadataCache(client, ttl=0.0)  # type: ignore[arg-type] — force re-refresh each call
        await cache.ensure_fresh()
        assert "light.a" in cache._entities
        # Force another refresh by waiting past TTL (ttl=0 means always re-fetch)
        await cache.ensure_fresh()
        # Old cache retained
        assert "light.a" in cache._entities
        # _fetched_at advanced (is finite, not the prior moment)
        assert cache._fetched_at > 0.0

    async def test_initial_fetched_at_triggers_refresh(self):
        """With _fetched_at=0.0, the very first ensure_fresh must actually refresh."""
        client = RecordingClient(
            responses=[self._EMPTY, self._EMPTY, self._EMPTY]
        )
        cache = EntityMetadataCache(client, ttl=60.0)  # type: ignore[arg-type]
        assert cache._fetched_at == 0.0
        await cache.ensure_fresh()
        assert client.call_count == 3

    async def test_label_registry_names_expanded_into_meta_labels(self):
        """The cache fetches label_registry and expands label IDs on entities
        to include their human names, enabling policy rules to reference
        labels by either spelling."""
        client = RecordingClient(
            responses=[
                # entity_registry: one entity with a label ID
                {"success": True, "result": [{
                    "entity_id": "light.alice_bed",
                    "area_id": None,
                    "labels": ["owner_alice"],  # HA stores the slug/id
                    "device_id": None,
                }]},
                self._EMPTY,  # device_registry
                # label_registry: maps id -> human name
                {"success": True, "result": [
                    {"label_id": "owner_alice", "name": "owner:alice"},
                ]},
            ]
        )
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        resolved = cache._entities["light.alice_bed"]
        assert resolved.labels == {"owner_alice", "owner:alice"}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestYamlLoading:
    def test_loads_minimal_allowlist_policy(self, tmp_path: Path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "version: 1\n"
            "entities:\n"
            "  allow:\n"
            '    - entity_globs: ["light.alice_*"]\n'
        )
        cfg = load_policy_from_file(policy_file)
        assert cfg.default_action == "deny"  # default
        assert cfg.readonly_mode is False
        assert len(cfg.entities.allow) == 1
        assert cfg.entities.allow[0].entity_globs == ["light.alice_*"]

    def test_loads_full_policy(self, tmp_path: Path):
        policy_file = tmp_path / "alice.yaml"
        policy_file.write_text(
            "version: 1\n"
            "default_action: deny\n"
            "readonly_mode: false\n"
            "tools:\n"
            '  disabled_names: ["ha_backup_create"]\n'
            '  disabled_tags: ["HACS"]\n'
            "entities:\n"
            "  allow:\n"
            '    - labels: ["owner:alice"]\n'
            '    - areas: ["alice_bedroom"]\n'
            "  deny:\n"
            '    - entity_globs: ["lock.*"]\n'
            "  write_allow:\n"
            '    - labels: ["owner:alice"]\n'
            "  write_deny:\n"
            '    - entity_globs: ["climate.whole_house_*"]\n'
        )
        cfg = load_policy_from_file(policy_file)
        assert "ha_backup_create" in cfg.tools.disabled_names
        assert "HACS" in cfg.tools.disabled_tags
        assert len(cfg.entities.allow) == 2
        assert len(cfg.entities.deny) == 1
        assert cfg.entities.write_allow is not None
        assert len(cfg.entities.write_allow) == 1
        assert len(cfg.entities.write_deny) == 1

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(PolicyLoadError, match="not found"):
            load_policy_from_file(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path):
        policy_file = tmp_path / "bad.yaml"
        policy_file.write_text("version: 1\nentities: {unclosed\n")
        with pytest.raises(PolicyLoadError, match="malformed YAML"):
            load_policy_from_file(policy_file)

    def test_unknown_field_raises(self, tmp_path: Path):
        policy_file = tmp_path / "unknown.yaml"
        policy_file.write_text("version: 1\nbogus_field: true\n")
        with pytest.raises(PolicyLoadError, match="validation"):
            load_policy_from_file(policy_file)

    def test_invalid_default_action_raises(self, tmp_path: Path):
        policy_file = tmp_path / "invalid.yaml"
        policy_file.write_text("version: 1\ndefault_action: maybe\n")
        with pytest.raises(PolicyLoadError, match="validation"):
            load_policy_from_file(policy_file)

    def test_non_mapping_top_level_raises(self, tmp_path: Path):
        policy_file = tmp_path / "list.yaml"
        policy_file.write_text("- just\n- a\n- list\n")
        with pytest.raises(PolicyLoadError, match="mapping"):
            load_policy_from_file(policy_file)

    def test_empty_file_loads_defaults(self, tmp_path: Path):
        """An empty YAML file is equivalent to all defaults."""
        policy_file = tmp_path / "empty.yaml"
        policy_file.write_text("")
        cfg = load_policy_from_file(policy_file)
        assert cfg.default_action == "deny"
        assert cfg.entities.allow == []
