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

from .conftest import offline_client


class FakeCache(EntityMetadataCache):
    """Test double that bypasses websocket fetches.

    - Skips EntityMetadataCache.__init__ (no client needed).
    - Presets _fetched_at to +inf so ensure_fresh() and the base TTL check
      always short-circuit ("never expire"), letting each test control the
      cache contents directly.
    - If EntityMetadataCache gains new required instance state, update this
      constructor — tests will surface the breakage.
    """

    def __init__(
        self,
        entities: dict[str, EntityMeta] | None = None,
        *,
        area_labels: dict[str, set[str]] | None = None,
        state_only: set[str] | None = None,
    ):
        import asyncio

        self._entities = entities or {}
        self._area_labels = area_labels or {}
        self._area_names: dict[str, str] = {}
        # Entities with a state but no registry entry (the live
        # GET /api/states/<id> lookup answers True for these only).
        self._state_only = state_only or set()
        self._fetched_at = float("inf")
        self._lock = asyncio.Lock()

    async def ensure_fresh(self) -> None:  # override — no-op
        return

    async def _refresh(self) -> None:  # pragma: no cover — never called
        return

    async def _state_exists(self, entity_id: str) -> bool | None:
        return entity_id in self._state_only


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
    cfg: PolicyConfig,
    entities: dict[str, EntityMeta] | None = None,
    **cache_kwargs: Any,
) -> AccessPolicy:
    return AccessPolicy(cfg, FakeCache(entities, **cache_kwargs))


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
# Targeted service calls: cross-domain rule + allow_targeted (hi-hx8.52)
# ---------------------------------------------------------------------------


class TestTargetedServiceConfig:
    """Schema of services.allow_cross_domain / services.allow_targeted."""

    def test_defaults(self):
        s = ServicesPolicy()
        assert s.allow_cross_domain == []
        assert s.allow_targeted is None

    def test_valid_entries_accepted(self):
        s = ServicesPolicy(
            allow_cross_domain=["music_assistant.*", "sonos.snapshot"],
            allow_targeted=["light.turn_on", "media_player.*"],
        )
        assert s.allow_cross_domain == ["music_assistant.*", "sonos.snapshot"]
        assert s.allow_targeted == ["light.turn_on", "media_player.*"]
        # Empty allow_targeted is valid and distinct from None (unset).
        assert ServicesPolicy(allow_targeted=[]).allow_targeted == []

    @pytest.mark.parametrize("field", ["allow_cross_domain", "allow_targeted"])
    @pytest.mark.parametrize(
        "bad",
        ["music_assistant", "Light.turn_on", "light.turn*", "light.", ".turn_on",
         "light.turn_on.extra", "*.*", "*"],
    )
    def test_invalid_entries_rejected(self, field: str, bad: str):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=f"services.{field}"):
            ServicesPolicy(**{field: [bad]})

    def test_unknown_services_key_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ServicesPolicy.model_validate({"allow_crossdomain": ["sonos.*"]})

    def test_yaml_roundtrip_and_unknown_key(self, tmp_path: Path):
        good = tmp_path / "good.yaml"
        good.write_text(
            "version: 1\n"
            "services:\n"
            "  allow_cross_domain: ['music_assistant.*']\n"
            "  allow_targeted: ['light.turn_on', 'music_assistant.play_media']\n"
        )
        cfg = load_policy_from_file(good)
        assert cfg.services.allow_cross_domain == ["music_assistant.*"]
        assert cfg.services.allow_targeted == [
            "light.turn_on", "music_assistant.play_media",
        ]

        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1\nservices:\n  allow_target: ['light.turn_on']\n")
        with pytest.raises(PolicyLoadError, match="validation"):
            load_policy_from_file(bad)

        bad_entry = tmp_path / "bad_entry.yaml"
        bad_entry.write_text("version: 1\nservices:\n  allow_targeted: ['light']\n")
        with pytest.raises(PolicyLoadError, match="allow_targeted"):
            load_policy_from_file(bad_entry)


class TestTargetedServiceDenial:
    """AccessPolicy.targeted_service_denial — the pure service-level gate."""

    def test_same_domain_allowed(self):
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial("light", "turn_on", ["light.a"]) is None

    def test_cross_domain_denied_by_default(self):
        p = make_policy(PolicyConfig())
        reason = p.targeted_service_denial("script", "front_door_open", ["light.a"])
        assert reason is not None
        assert "cross-domain" in reason
        assert "script.front_door_open" in reason
        assert "allow_cross_domain" in reason

    def test_mixed_targets_denied(self):
        """EVERY target must be in the service's domain, not just one."""
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial(
            "light", "turn_on", ["light.a", "switch.b"]
        ) is not None

    def test_id_without_domain_denied(self):
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial("light", "turn_on", ["light"]) is not None
        assert p.targeted_service_denial("light", "turn_on", ["none"]) is not None

    def test_domain_prefix_is_not_a_domain_match(self):
        """``light_extra.x`` is not in the ``light`` domain."""
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial(
            "light", "turn_on", ["light_extra.x"]
        ) is not None

    @pytest.mark.parametrize(
        "service", ["turn_on", "turn_off", "toggle", "update_entity"]
    )
    def test_homeassistant_builtins_exempt(self, service: str):
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial(
            "homeassistant", service, ["light.a", "switch.b"]
        ) is None

    def test_other_homeassistant_services_not_exempt(self):
        p = make_policy(PolicyConfig())
        assert p.targeted_service_denial(
            "homeassistant", "reload_config_entry", ["light.a"]
        ) is not None

    def test_allow_cross_domain_exact_and_wildcard(self):
        p = make_policy(PolicyConfig(services=ServicesPolicy(
            allow_cross_domain=["music_assistant.*", "sonos.snapshot"],
        )))
        assert p.targeted_service_denial(
            "music_assistant", "play_media", ["media_player.x"]
        ) is None
        assert p.targeted_service_denial(
            "sonos", "snapshot", ["media_player.x"]
        ) is None
        assert p.targeted_service_denial(
            "sonos", "restore", ["media_player.x"]
        ) is not None

    def test_allow_targeted_restricts_services(self):
        p = make_policy(PolicyConfig(services=ServicesPolicy(
            allow_targeted=["light.turn_on", "light.turn_off"],
        )))
        assert p.targeted_service_denial("light", "turn_on", ["light.a"]) is None
        reason = p.targeted_service_denial("light", "toggle", ["light.a"])
        assert reason is not None and "allow_targeted" in reason
        # homeassistant.* is NOT implied by allow_targeted.
        assert p.targeted_service_denial(
            "homeassistant", "turn_off", ["light.a"]
        ) is not None

    def test_allow_targeted_empty_blocks_every_targeted_call(self):
        p = make_policy(PolicyConfig(services=ServicesPolicy(allow_targeted=[])))
        assert p.targeted_service_denial("light", "turn_on", ["light.a"]) is not None

    def test_allow_targeted_does_not_bypass_cross_domain(self):
        p = make_policy(PolicyConfig(services=ServicesPolicy(
            allow_targeted=["music_assistant.*"],
        )))
        reason = p.targeted_service_denial(
            "music_assistant", "play_media", ["media_player.x"]
        )
        assert reason is not None and "cross-domain" in reason

    def test_case_insensitive_like_ha(self):
        """HA lowercases domain/service, so the gate matches on lowercase."""
        p = make_policy(PolicyConfig(services=ServicesPolicy(
            allow_targeted=["light.turn_on"],
        )))
        assert p.targeted_service_denial("LIGHT", "Turn_On", ["light.a"]) is None
        assert p.targeted_service_denial(
            "Script", "front_door_open", ["light.a"]
        ) is not None


# Entities for the client-level enforcement tests below.
_SVC_ENTITIES = {
    "light.kitchen": meta(
        "light.kitchen", area_id="kitchen", labels={"lights"}, device_id="dev_k"
    ),
    "light.kitchen_2": meta(
        "light.kitchen_2", area_id="kitchen", labels={"lights"}, device_id="dev_k"
    ),
    "media_player.living": meta("media_player.living", area_id="living"),
    "switch.porch": meta("switch.porch", area_id="porch"),
    "lock.front_door": meta("lock.front_door", area_id="hall"),
    "script.kids_bedtime": meta("script.kids_bedtime"),
    "script.front_door_open": meta("script.front_door_open"),
    # Added for the service-data / unknown-target / label-area cases:
    "media_player.kitchen": meta("media_player.kitchen", area_id="kitchen_nook"),
    "media_player.bedroom": meta("media_player.bedroom", area_id="bedroom"),
    "camera.front": meta("camera.front"),
    "tts.piper": meta("tts.piper"),
    "automation.kids_lights": meta("automation.kids_lights"),
    "scene.kids_evening": meta("scene.kids_evening"),
    "light.landing": meta("light.landing", area_id="landing"),
}

_SVC_ALLOW = [
    EntityRule(
        entity_globs=[
            "light.*",
            "media_player.living",
            "media_player.kitchen",
            "switch.porch",
            "script.kids_bedtime",
            "camera.front",
            "tts.piper",
            "automation.kids_lights",
            "scene.kids_evening",
        ]
    )
]


def _svc_client(
    services: ServicesPolicy | None = None,
    *,
    area_labels: dict[str, set[str]] | None = None,
    state_only: set[str] | None = None,
    **cfg_kwargs: Any,
):
    """HomeAssistantClient with a policy and a mocked transport.

    ``_request`` is an AsyncMock, so a test can assert that a denied call
    never reached Home Assistant. Every ``_SVC_ENTITIES`` entry is a known
    (registry) entity; ``state_only`` adds entities that only have a state.
    """
    cfg_kwargs.setdefault("entities", EntitiesPolicy(allow=_SVC_ALLOW))
    cfg = PolicyConfig(services=services or ServicesPolicy(), **cfg_kwargs)
    return offline_client(
        make_policy(
            cfg, dict(_SVC_ENTITIES), area_labels=area_labels, state_only=state_only
        )
    )


async def _assert_denied(
    client: Any, domain: str, service: str, data: dict[str, Any],
    *, reason_contains: str | None = None,
) -> dict[str, Any]:
    """Call the service, assert ACCESS_DENIED, and that HA was never called."""
    import json

    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await client.call_service(domain, service, data)
    payload = json.loads(str(exc.value))
    assert payload["error"]["code"] == "ACCESS_DENIED"
    if reason_contains is not None:
        assert reason_contains in payload["error"]["details"]
    client._request.assert_not_awaited()
    return payload


async def _assert_allowed(
    client: Any, domain: str, service: str, data: dict[str, Any]
) -> None:
    await client.call_service(domain, service, data)
    client._request.assert_awaited_once()
    assert client._request.await_args.args[:2] == (
        "POST", f"/services/{domain}/{service}"
    )


class TestServiceCallEnforcement:
    """End-to-end (offline) through HomeAssistantClient.call_service."""

    # -- the service name is a slug, never a path --------------------------
    @pytest.mark.parametrize(
        ("domain", "service"),
        [
            ("light", "../script/front_door_open"),
            ("light", "turn_on/../../script/front_door_open"),
            ("light", "..%2Fscript%2Ffront_door_open"),
            ("light", "turn_on?x=1"),
            ("../script", "front_door_open"),
            ("light", ""),
        ],
    )
    async def test_path_like_service_names_are_refused(
        self, domain: str, service: str
    ):
        """httpx collapses "..": without this, the gate saw light.* and HA ran
        script.front_door_open."""
        await _assert_denied(
            _svc_client(), domain, service, {"entity_id": "light.kitchen"},
            reason_contains="slug",
        )

    async def test_service_name_is_normalised_before_the_gate(self):
        client = _svc_client()
        await client.call_service(" Light ", "TURN_ON", {"entity_id": "light.kitchen"})
        assert client._request.await_args.args[:2] == ("POST", "/services/light/turn_on")

    @pytest.mark.parametrize("segment", ["", ".", ".."])
    def test_dot_segments_are_refused(self, segment: str):
        """quote() leaves dots alone and httpx collapses them: /states/.. -> /api."""
        from ha_mcp.client.rest_client import _path_segment

        with pytest.raises(ValueError):
            _path_segment(segment)
        assert _path_segment("light.kitchen") == "light.kitchen"

    async def test_state_and_event_paths_are_quoted(self):
        from unittest.mock import AsyncMock

        client = offline_client(request=AsyncMock(return_value={}))
        await client.get_entity_state("light.x/../../services/script/front_door_open")
        assert client._request.await_args.args[1] == (
            "/states/light.x%2F..%2F..%2Fservices%2Fscript%2Ffront_door_open"
        )
        await client.fire_event("x/../../services/script/front_door_open")
        assert client._request.await_args.args[1].startswith("/events/x%2F..")

    async def test_caller_supplied_path_segments_are_quoted(self):
        from unittest.mock import AsyncMock

        client = offline_client(request=AsyncMock(return_value={}))
        client.send_websocket_message = AsyncMock(  # type: ignore[method-assign]
            return_value={"success": False}
        )
        evil = "x/../../services/script/front_door_open"
        quoted = "x%2F..%2F..%2Fservices%2Fscript%2Ffront_door_open"
        calls = [
            (lambda: client.get_history(entity_id=evil), "/history/period/"),
            (lambda: client.get_automation_config(evil), "/config/automation/config/"),
            (lambda: client.submit_config_flow_step(evil, {}),
             "/config/config_entries/flow/"),
            (lambda: client.abort_config_flow(evil), "/config/config_entries/flow/"),
            (lambda: client.submit_options_flow_step(evil, {}),
             "/config/config_entries/options/flow/"),
            (lambda: client.abort_options_flow(evil),
             "/config/config_entries/options/flow/"),
            (lambda: client.delete_config_entry(evil), "/config/config_entries/entry/"),
            (lambda: client.get_script_config(evil), "config/script/config/"),
            (lambda: client.delete_script_config(evil), "config/script/config/"),
        ]
        for call, prefix in calls:
            await call()
            assert client._request.await_args.args[1] == prefix + quoted

    @pytest.mark.parametrize(
        ("domain", "service", "data"),
        [
            ("script", "front_door_open", {"entity_id": "light.kitchen"}),
            ("shell_command", "x", {"entity_id": "light.kitchen"}),
            ("rest_command", "x", {"entity_id": "light.kitchen"}),
            ("hassio", "addon_restart",
             {"addon": "core_ssh", "entity_id": "light.kitchen"}),
            ("notify", "x",
             {"message": "hi", "target": {"entity_id": "light.kitchen"}}),
        ],
    )
    async def test_cross_domain_denied(
        self, domain: str, service: str, data: dict[str, Any]
    ):
        client = _svc_client()
        payload = await _assert_denied(
            client, domain, service, data, reason_contains="cross-domain"
        )
        assert f"{domain}.{service}" in payload["error"]["message"]

    @pytest.mark.parametrize(
        "data",
        [
            {"target": {"entity_id": "light.kitchen"}},
            {"target": {"entity_id": ["light.kitchen"]}},
            {"device_id": "dev_k"},
            {"area_id": "kitchen"},
            {"label_id": "lights"},
            {"target": {"device_id": ["dev_k"]}},
            {"target": {"area_id": "kitchen"}},
            {"target": {"label_id": "lights"}},
        ],
    )
    async def test_cross_domain_denied_for_every_selector_shape(
        self, data: dict[str, Any]
    ):
        """Script aimed at allowed lights via every selector → denied."""
        client = _svc_client()
        await _assert_denied(
            client, "script", "front_door_open", data,
            reason_contains="cross-domain",
        )

    async def test_same_domain_allowed(self):
        await _assert_allowed(
            _svc_client(), "light", "turn_on", {"entity_id": "light.kitchen"}
        )

    async def test_same_domain_area_target_allowed(self):
        await _assert_allowed(
            _svc_client(), "light", "turn_off", {"target": {"area_id": "kitchen"}}
        )

    @pytest.mark.parametrize(
        "service", ["turn_on", "turn_off", "toggle", "update_entity"]
    )
    async def test_homeassistant_builtins_allowed(self, service: str):
        await _assert_allowed(
            _svc_client(), "homeassistant", service, {"entity_id": "light.kitchen"}
        )

    async def test_homeassistant_builtin_still_entity_checked(self):
        """The exemption is from the domain rule only, not the entity check."""
        payload = await _assert_denied(
            _svc_client(), "homeassistant", "turn_off",
            {"entity_id": ["light.kitchen", "lock.front_door"]},
        )
        assert payload["entity_id"] == "lock.front_door"

    async def test_music_assistant_denied_by_default(self):
        await _assert_denied(
            _svc_client(), "music_assistant", "play_media",
            {"entity_id": "media_player.living", "media_id": "x"},
            reason_contains="cross-domain",
        )

    @pytest.mark.parametrize(
        "entry", ["music_assistant.*", "music_assistant.play_media"]
    )
    async def test_music_assistant_allowed_with_cross_domain(self, entry: str):
        client = _svc_client(ServicesPolicy(allow_cross_domain=[entry]))
        await _assert_allowed(
            client, "music_assistant", "play_media",
            {"entity_id": "media_player.living", "media_id": "x"},
        )

    async def test_cross_domain_allowlist_still_entity_checked(self):
        client = _svc_client(ServicesPolicy(allow_cross_domain=["script.*"]))
        await _assert_denied(
            client, "script", "front_door_open", {"entity_id": "lock.front_door"}
        )

    # -- allow_targeted -----------------------------------------------------

    @staticmethod
    def _targeted_client() -> Any:
        return _svc_client(
            ServicesPolicy(allow_targeted=["light.turn_on", "light.turn_off"])
        )

    async def test_allow_targeted_permits_listed(self):
        await _assert_allowed(
            self._targeted_client(), "light", "turn_on",
            {"entity_id": "light.kitchen"},
        )

    async def test_allow_targeted_denies_unlisted_same_domain(self):
        await _assert_denied(
            self._targeted_client(), "light", "toggle",
            {"entity_id": "light.kitchen"}, reason_contains="allow_targeted",
        )

    async def test_allow_targeted_denies_homeassistant_builtin(self):
        await _assert_denied(
            self._targeted_client(), "homeassistant", "turn_off",
            {"entity_id": "light.kitchen"}, reason_contains="allow_targeted",
        )

    async def test_allow_targeted_denies_integration_service(self):
        await _assert_denied(
            self._targeted_client(), "sonos", "update_alarm",
            {"entity_id": "switch.porch", "alarm_id": 1},
        )

    async def test_allow_targeted_and_cross_domain_combine(self):
        client = _svc_client(ServicesPolicy(
            allow_targeted=["sonos.*"], allow_cross_domain=["sonos.update_alarm"],
        ))
        await _assert_allowed(
            client, "sonos", "update_alarm", {"entity_id": "switch.porch"}
        )

    # -- script.<name> runs itself, whatever it targets ---------------------

    async def test_script_service_targeting_writable_script_still_checks_itself(self):
        """script.front_door_open aimed at a writable script → still denied."""
        payload = await _assert_denied(
            _svc_client(), "script", "front_door_open",
            {"entity_id": "script.kids_bedtime"},
        )
        assert payload["entity_id"] == "script.front_door_open"

    async def test_script_service_on_itself_allowed(self):
        await _assert_allowed(
            _svc_client(), "script", "kids_bedtime",
            {"entity_id": "script.kids_bedtime"},
        )

    async def test_script_turn_on_checks_only_targets(self):
        await _assert_allowed(
            _svc_client(), "script", "turn_on", {"entity_id": "script.kids_bedtime"}
        )

    # -- entity_id canonicalization (HA splits CSV + lowercases) ------------

    async def test_csv_entity_id_string_is_split(self):
        """'light.kitchen, lock.front_door' is two targets in HA, not one."""
        client = _svc_client(
            entities=EntitiesPolicy(allow=[EntityRule(entity_globs=["light.kitchen*"])])
        )
        await _assert_denied(
            client, "homeassistant", "turn_off",
            {"entity_id": "light.kitchen, lock.front_door"},
        )

    async def test_uppercase_entity_id_hits_deny_rules(self):
        client = _svc_client(
            default_action="allow",
            entities=EntitiesPolicy(write_deny=[EntityRule(entity_globs=["lock.*"])]),
        )
        await _assert_denied(
            client, "lock", "unlock", {"entity_id": "LOCK.FRONT_DOOR"}
        )

    async def test_uppercase_all_wildcard_denied(self):
        client = _svc_client(default_action="allow")
        await _assert_denied(
            client, "homeassistant", "turn_on", {"entity_id": "ALL"},
            reason_contains="'all'",
        )

    # -- unchanged behaviour ------------------------------------------------

    async def test_readonly_mode_denies_targeted(self):
        client = _svc_client(readonly_mode=True)
        await _assert_denied(
            client, "light", "turn_on", {"entity_id": "light.kitchen"}
        )

    async def test_targetless_denied_by_default(self):
        await _assert_denied(
            _svc_client(), "persistent_notification", "create", {"message": "x"},
            reason_contains="allow_targetless",
        )

    async def test_targetless_governed_by_allow_targetless_only(self):
        """allow_targeted / the domain rule don't touch targetless calls."""
        client = _svc_client(ServicesPolicy(
            allow_targetless=["persistent_notification.create", "script.front_door_open"],
            allow_targeted=[],
        ))
        await client.call_service("persistent_notification", "create", {"message": "x"})
        await client.call_service("script", "front_door_open", {})
        assert client._request.await_count == 2

    async def test_targetless_allowlist_does_not_cover_targeted(self):
        """allow_targetless doesn't let the same service be aimed cross-domain."""
        client = _svc_client(ServicesPolicy(allow_targetless=["script.*"]))
        await _assert_denied(
            client, "script", "front_door_open", {"entity_id": "light.kitchen"},
            reason_contains="cross-domain",
        )


class TestWebSocketServiceCallEnforcement:
    """WS ``call_service`` goes through the same gate as REST."""

    @staticmethod
    def _ws_client() -> tuple[Any, Any]:
        from unittest.mock import AsyncMock

        client = _svc_client()
        ws = AsyncMock()
        ws.send_command = AsyncMock(return_value={"success": True})
        return client, ws

    async def test_ws_call_service_cross_domain_denied(self):
        from unittest.mock import AsyncMock, patch

        from fastmcp.exceptions import ToolError

        client, ws = self._ws_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ), pytest.raises(ToolError, match="cross-domain"):
            await client.send_websocket_message({
                "type": "call_service",
                "domain": "script",
                "service": "front_door_open",
                "target": {"entity_id": "light.kitchen"},
            })
        ws.send_command.assert_not_awaited()

    async def test_ws_call_service_same_domain_allowed(self):
        from unittest.mock import AsyncMock, patch

        client, ws = self._ws_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ):
            await client.send_websocket_message({
                "type": "call_service",
                "domain": "light",
                "service": "turn_on",
                "service_data": {"brightness": 10},
                "target": {"entity_id": "light.kitchen"},
            })
        ws.send_command.assert_awaited_once()

    @pytest.mark.parametrize(
        ("domain", "service"),
        [("light", "../script/front_door_open"), ("../script", "front_door_open")],
    )
    async def test_ws_call_service_path_like_service_name_refused(
        self, domain: str, service: str
    ):
        from unittest.mock import AsyncMock, patch

        from fastmcp.exceptions import ToolError

        client, ws = self._ws_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ), pytest.raises(ToolError, match="slug"):
            await client.send_websocket_message({
                "type": "call_service",
                "domain": domain,
                "service": service,
                "target": {"entity_id": "light.kitchen"},
            })
        ws.send_command.assert_not_awaited()

    async def test_ws_call_service_name_normalised_before_the_gate(self):
        """' Light '/'TURN_ON' is gated as light.turn_on, like HA resolves it."""
        from unittest.mock import AsyncMock, patch

        client, ws = self._ws_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ):
            await client.send_websocket_message({
                "type": "call_service",
                "domain": " Light ",
                "service": "TURN_ON",
                "target": {"entity_id": "light.kitchen"},
            })
        ws.send_command.assert_awaited_once()

    async def test_ws_execute_script_denied(self):
        from unittest.mock import AsyncMock, patch

        from fastmcp.exceptions import ToolError

        client, ws = self._ws_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ), pytest.raises(ToolError, match="ACCESS_DENIED"):
            await client.send_websocket_message({
                "type": "execute_script",
                "sequence": [{"action": "lock.unlock",
                              "target": {"entity_id": "lock.front_door"}}],
            })
        ws.send_command.assert_not_awaited()


class TestServiceDataEntities:
    """Entity ids named in service DATA (not the target) are write-checked."""

    _BEDROOM = "media_player.bedroom"  # known, not writable

    @pytest.mark.parametrize(
        ("domain", "service", "data", "services", "key"),
        [
            ("media_player", "join",
             {"entity_id": "media_player.living", "group_members": [_BEDROOM]},
             None, "group_members[0]"),
            ("tts", "speak",
             {"entity_id": "tts.piper", "media_player_entity_id": _BEDROOM,
              "message": "hi"},
             None, "media_player_entity_id"),
            ("camera", "play_stream",
             {"entity_id": "camera.front", "media_player": _BEDROOM},
             None, "media_player"),
            ("music_assistant", "transfer_queue",
             {"entity_id": "media_player.living", "source_player": _BEDROOM},
             ServicesPolicy(allow_cross_domain=["music_assistant.*"]),
             "source_player"),
            # entities is a mapping KEYED by entity id
            ("scene", "apply", {"entities": {_BEDROOM: "playing"}},
             ServicesPolicy(allow_targetless=["scene.apply"]), "entities"),
            ("scene", "create",
             {"scene_id": "x", "entities": {"lock.front_door": {"state": "unlocked"}}},
             ServicesPolicy(allow_targetless=["scene.create"]), "entities"),
            ("scene", "create",
             {"scene_id": "x", "snapshot_entities": ["lock.front_door"]},
             ServicesPolicy(allow_targetless=["scene.create"]),
             "snapshot_entities[0]"),
            ("group", "set", {"object_id": "kids", "entities": ["lock.front_door"]},
             ServicesPolicy(allow_targetless=["group.set"]), "entities[0]"),
            ("group", "set",
             {"object_id": "kids", "add_entities": ["light.kitchen", _BEDROOM]},
             ServicesPolicy(allow_targetless=["group.set"]), "add_entities[1]"),
            ("group", "set",
             {"object_id": "kids", "remove_entities": "lock.front_door"},
             ServicesPolicy(allow_targetless=["group.set"]), "remove_entities"),
            # nested script variables
            ("script", "turn_on",
             {"entity_id": "script.kids_bedtime",
              "variables": {"who": {"lock": "lock.front_door"}}},
             None, "variables.who.lock"),
            # non-selector keys under target are data too
            ("media_player", "join",
             {"target": {"entity_id": "media_player.living",
                         "group_members": [_BEDROOM]}},
             None, "target.group_members[0]"),
            # canonicalised like entity_id: CSV split + lowercase
            ("media_player", "join",
             {"entity_id": "media_player.living",
              "group_members": "media_player.kitchen, MEDIA_PLAYER.BEDROOM"},
             None, "group_members"),
        ],
    )
    async def test_unwritable_entity_in_data_denied(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        services: ServicesPolicy | None,
        key: str,
    ):
        payload = await _assert_denied(
            _svc_client(services), domain, service, data,
            reason_contains=f"key '{key}'",
        )
        assert payload["entity_id"] in {self._BEDROOM, "lock.front_door"}

    async def test_state_only_entity_in_data_denied(self):
        """An entity with a state but no registry entry still counts."""
        await _assert_denied(
            _svc_client(state_only={"media_player.guest"}),
            "media_player", "join",
            {"entity_id": "media_player.living",
             "group_members": ["media_player.guest"]},
            reason_contains="group_members",
        )

    @pytest.mark.parametrize(
        ("domain", "service", "data", "services"),
        [
            ("media_player", "join",
             {"entity_id": "media_player.living",
              "group_members": ["media_player.kitchen"]}, None),
            ("tts", "speak",
             {"entity_id": "tts.piper",
              "media_player_entity_id": "media_player.living",
              "message": "lock.front_door is not an entity id here, the whole "
                         "string is the message"}, None),
            # looks like an id but names no entity → just data
            ("media_player", "play_media",
             {"entity_id": "media_player.living",
              "media_content_id": "spotify.track",
              "media_content_type": "music"}, None),
            ("scene", "apply", {"entities": {"light.kitchen": "on"}},
             ServicesPolicy(allow_targetless=["scene.apply"])),
            ("group", "set",
             {"object_id": "kids", "entities": ["light.kitchen", "light.landing"]},
             ServicesPolicy(allow_targetless=["group.set"])),
        ],
    )
    async def test_writable_or_non_entity_data_allowed(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        services: ServicesPolicy | None,
    ):
        await _assert_allowed(_svc_client(services), domain, service, data)

    async def test_unverifiable_id_in_data_fails_closed(self):
        """A failed state lookup (None) counts as a known entity."""
        from unittest.mock import AsyncMock

        client = _svc_client()
        client.policy.cache._state_exists = AsyncMock(return_value=None)
        await _assert_denied(
            client, "media_player", "play_media",
            {"entity_id": "media_player.living",
             "media_content_id": "spotify.track"},
            reason_contains="media_content_id",
        )

    async def test_writable_data_entity_needs_no_lookup(self):
        from unittest.mock import AsyncMock

        client = _svc_client()
        client.policy.cache._state_exists = AsyncMock(return_value=None)
        await _assert_allowed(
            client, "media_player", "join",
            {"entity_id": "media_player.living",
             "group_members": ["media_player.kitchen"]},
        )
        client.policy.cache._state_exists.assert_not_awaited()

    async def test_targetless_allowed_call_data_checked(self):
        """allow_targetless does not exempt the data from the entity check."""
        await _assert_denied(
            _svc_client(ServicesPolicy(allow_targetless=["persistent_notification.*"])),
            "persistent_notification", "create",
            {"message": "x", "notification_id": "lock.front_door"},
            reason_contains="notification_id",
        )

    async def test_ws_call_service_data_checked(self):
        from unittest.mock import AsyncMock, patch

        from fastmcp.exceptions import ToolError

        client = _svc_client()
        ws = AsyncMock()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ), pytest.raises(ToolError, match="group_members"):
            await client.send_websocket_message({
                "type": "call_service",
                "domain": "media_player",
                "service": "join",
                "service_data": {"group_members": [self._BEDROOM]},
                "target": {"entity_id": "media_player.living"},
            })
        ws.send_command.assert_not_awaited()


class TestLabelTargetsExpandAreas:
    """A label_id target also covers every entity of every labelled AREA."""

    async def test_label_on_area_with_unwritable_entity_denied(self):
        client = _svc_client(
            area_labels={"landing": {"kids"}},
            entities=EntitiesPolicy(
                allow=_SVC_ALLOW,
                write_deny=[EntityRule(entity_globs=["light.landing"])],
            ),
        )
        payload = await _assert_denied(
            client, "light", "turn_off", {"label_id": "kids"}
        )
        assert payload["entity_id"] == "light.landing"

    async def test_label_on_mixed_area_denied_cross_domain(self):
        """Area 'hall' holds the lock: a light service can't act on it."""
        await _assert_denied(
            _svc_client(area_labels={"hall": {"kids"}}),
            "light", "turn_off", {"target": {"label_id": ["kids"]}},
            reason_contains="lock.front_door",
        )

    async def test_label_on_writable_area_and_entities_allowed(self):
        client = _svc_client(area_labels={"landing": {"cozy"}, "hall": set()})
        await _assert_allowed(client, "light", "turn_off", {"label_id": "cozy"})
        await _assert_allowed(
            _svc_client(), "light", "turn_off", {"label_id": "lights"}
        )

    async def test_label_unresolvable_without_area_registry(self):
        client = _svc_client()
        client.policy.cache._area_labels = None
        await _assert_denied(
            client, "light", "turn_off", {"label_id": "lights"},
            reason_contains="area registry",
        )


class TestHomeassistantProxyExemption:
    """homeassistant.turn_on/off/toggle may not proxy to script/automation/scene."""

    @pytest.mark.parametrize(
        ("service", "target"),
        [
            ("turn_on", "script.kids_bedtime"),
            ("turn_off", "automation.kids_lights"),
            ("toggle", "automation.kids_lights"),
            ("turn_on", "scene.kids_evening"),
            ("turn_on", ["light.kitchen", "script.kids_bedtime"]),
        ],
    )
    async def test_proxy_to_runnable_domain_denied(
        self, service: str, target: str | list[str]
    ):
        await _assert_denied(
            _svc_client(), "homeassistant", service,
            {"entity_id": target, "variables": {"x": 1}},
            reason_contains="script/automation/scene",
        )

    async def test_update_entity_keeps_exemption(self):
        await _assert_allowed(
            _svc_client(), "homeassistant", "update_entity",
            {"entity_id": ["script.kids_bedtime", "automation.kids_lights"]},
        )

    @pytest.mark.parametrize(
        ("domain", "target"),
        [("script", "script.kids_bedtime"),
         ("automation", "automation.kids_lights"),
         ("scene", "scene.kids_evening")],
    )
    async def test_own_domain_service_allowed(self, domain: str, target: str):
        await _assert_allowed(
            _svc_client(), domain, "turn_on", {"entity_id": target}
        )

    async def test_listing_in_allow_targeted_does_not_reopen_it(self):
        client = _svc_client(ServicesPolicy(allow_targeted=["homeassistant.*"]))
        await _assert_denied(
            client, "homeassistant", "turn_on",
            {"entity_id": "script.kids_bedtime"},
            reason_contains="script/automation/scene",
        )

    async def test_explicit_allow_cross_domain_permits(self):
        client = _svc_client(
            ServicesPolicy(allow_cross_domain=["homeassistant.turn_on"])
        )
        await _assert_allowed(
            client, "homeassistant", "turn_on", {"entity_id": "script.kids_bedtime"}
        )


class TestUnknownTargets:
    """Every resolved target of a targeted call must be an entity HA knows."""

    @pytest.mark.parametrize(
        ("domain", "service", "target"),
        [
            ("shell_command", "x", "shell_command.fake"),
            ("light", "turn_on", "light.ghost"),
            ("homeassistant", "turn_on", ["light.kitchen", "switch.ghost"]),
        ],
    )
    async def test_unknown_target_denied_under_default_allow(
        self, domain: str, service: str, target: str | list[str]
    ):
        await _assert_denied(
            _svc_client(default_action="allow"), domain, service,
            {"entity_id": target}, reason_contains="unknown target entity",
        )

    async def test_unknown_writable_by_glob_denied_under_default_deny(self):
        """light.* allows light.ghost, but there is no such entity."""
        await _assert_denied(
            _svc_client(), "light", "turn_on", {"entity_id": "light.ghost"},
            reason_contains="unknown target entity",
        )

    async def test_unwritable_unknown_gets_plain_denial(self):
        """The write check runs first: no existence oracle for denied ids."""
        payload = await _assert_denied(
            _svc_client(), "switch", "turn_on", {"entity_id": "switch.ghost"}
        )
        assert "unknown" not in payload["error"]["details"]

    async def test_state_only_target_is_known(self):
        await _assert_allowed(
            _svc_client(default_action="allow", state_only={"group.legacy"}),
            "group", "reload_members", {"entity_id": "group.legacy"},
        )

    async def test_unverifiable_target_denied(self):
        from unittest.mock import AsyncMock

        client = _svc_client(default_action="allow")
        client.policy.cache._state_exists = AsyncMock(return_value=None)
        await _assert_denied(
            client, "light", "turn_on", {"entity_id": "light.ghost"},
            reason_contains="could not verify",
        )

    async def test_known_targets_need_no_lookup(self):
        from unittest.mock import AsyncMock

        client = _svc_client(default_action="allow")
        client.policy.cache._state_exists = AsyncMock(return_value=None)
        await _assert_allowed(
            client, "light", "turn_on", {"entity_id": ["light.kitchen", "light.landing"]}
        )
        client.policy.cache._state_exists.assert_not_awaited()

    async def test_only_explicit_targets_are_checked_for_existence(self):
        """Device/area/label expansion comes from the registry: no lookup."""
        from unittest.mock import AsyncMock

        client = _svc_client()
        client.policy.cache.is_known_batch = AsyncMock(
            return_value={"light.landing": True}
        )
        await _assert_allowed(
            client, "light", "turn_off",
            {"entity_id": "light.landing", "area_id": "kitchen"},
        )
        client.policy.cache.is_known_batch.assert_awaited_once_with(["light.landing"])

    async def test_first_unknown_target_in_order_is_denied(self):
        """Every miss is looked up; the denial names the first unknown id."""
        client = _svc_client(
            default_action="allow", state_only={"light.b_state_only"}
        )
        payload = await _assert_denied(
            client, "light", "turn_on",
            {"entity_id": ["light.d_ghost", "light.b_state_only", "light.c_ghost"]},
            reason_contains="unknown target entity",
        )
        assert payload["entity_id"] == "light.c_ghost"


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

    async def test_expand_label_includes_labelled_area(self):
        """HA expands a label through areas too: every entity in the area."""
        p = self._policy_with_entities()
        p.cache._area_labels = {"alice_bedroom": {"owner:bob"}, "kitchen": set()}
        result = await p.expand_targets(label_ids=["owner:bob"])
        assert result == {"light.bob_bed", "light.alice_bed", "sensor.alice_temp"}

    async def test_expand_label_without_area_registry_raises(self):
        from ha_mcp.access_policy import UnresolvableTargetError

        p = self._policy_with_entities()
        p.cache._area_labels = None
        with pytest.raises(UnresolvableTargetError):
            await p.expand_targets(label_ids=["owner:bob"])
        # area/device targets don't need the area registry
        assert await p.expand_targets(area_ids=["kitchen"]) == {"light.kitchen"}


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
    """Minimal client answering registry fetches by message type.

    ``responses`` maps a registry ("entity", "device", "label", "area") to
    the responses for successive refreshes; a registry without (remaining)
    responses answers an empty success.
    """

    _EMPTY: ClassVar[dict[str, Any]] = {"success": True, "result": []}

    def __init__(self, responses: dict[str, list[Any]]):
        self._responses = {k: list(v) for k, v in responses.items()}
        self.call_count = 0

    async def send_websocket_message(self, msg: dict[str, Any]) -> Any:
        self.call_count += 1
        registry = msg["type"].split("/")[1].removesuffix("_registry")
        queue = self._responses.get(registry) or [self._EMPTY]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _registry(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "result": list(entries)}


def _entity(entity_id: str) -> dict[str, Any]:
    return {"entity_id": entity_id, "area_id": None, "labels": [], "device_id": None}


class TestCacheTTL:
    # Each _refresh() issues 4 WS calls (entity, device, label and area
    # registries).

    async def test_invalidate_forces_refresh(self):
        """invalidate() sets _fetched_at back to 0, so next ensure_fresh refetches."""
        # Two entities on first fetch, one on second fetch
        client = RecordingClient({
            "entity": [
                _registry(_entity("light.a"), _entity("light.b")),
                _registry(_entity("light.a")),
            ],
        })
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        assert set(cache._entities.keys()) == {"light.a", "light.b"}
        assert client.call_count == 4

        cache.invalidate()
        await cache.ensure_fresh()
        assert set(cache._entities.keys()) == {"light.a"}
        assert client.call_count == 8

    async def test_failed_refresh_keeps_previous_cache_but_marks_fetched(self):
        """When entity_registry fetch fails after prior success, keep old cache
        but DO advance _fetched_at so we don't stampede the backend on every call."""
        client = RecordingClient({
            "entity": [
                _registry(_entity("light.a")),
                # Second refresh: entity_registry fails
                RuntimeError("entity registry down"),
            ],
        })
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
        client = RecordingClient({})
        cache = EntityMetadataCache(client, ttl=60.0)  # type: ignore[arg-type]
        assert cache._fetched_at == 0.0
        await cache.ensure_fresh()
        assert client.call_count == 4

    async def test_label_registry_names_expanded_into_meta_labels(self):
        """The cache fetches label_registry and expands label IDs on entities
        to include their human names, enabling policy rules to reference
        labels by either spelling."""
        client = RecordingClient({
            # entity_registry: one entity with a label ID
            "entity": [_registry({
                "entity_id": "light.alice_bed",
                "area_id": None,
                "labels": ["owner_alice"],  # HA stores the slug/id
                "device_id": None,
            })],
            # label_registry: maps id -> human name
            "label": [_registry({"label_id": "owner_alice", "name": "owner:alice"})],
        })
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        resolved = cache._entities["light.alice_bed"]
        assert resolved.labels == {"owner_alice", "owner:alice"}

    async def test_area_labels_cached_separately(self):
        """Area labels come from area_registry, expanded to ids | names, and
        do NOT leak into the entity's own labels (rule matching unchanged)."""
        client = RecordingClient({
            "entity": [_registry({
                "entity_id": "light.kid", "area_id": "kid_room",
                "labels": [], "device_id": None,
            })],
            "label": [_registry({"label_id": "kids", "name": "Kids"})],
            "area": [_registry(
                {"area_id": "kid_room", "name": "Kid", "labels": ["kids"]},
                {"area_id": "hall", "name": "Hall"},
            )],
        })
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        assert cache.area_labels() == {"kid_room": {"kids", "Kids"}, "hall": set()}
        assert cache._entities["light.kid"].labels == set()

    async def test_area_registry_failure_keeps_unresolvable_state(self):
        """Area registry never fetched → area_labels() is None (fail closed)."""
        client = RecordingClient({"area": [RuntimeError("area registry down")]})
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        await cache.ensure_fresh()
        assert cache.area_labels() is None


class TestIsKnown:
    """EntityMetadataCache.is_known_batch: registry hit, else live state lookup."""

    @staticmethod
    def _cache(request_side_effect: Any = None) -> tuple[Any, Any]:
        from unittest.mock import AsyncMock

        client = RecordingClient({"entity": [_registry(_entity("light.a"))]})
        client._request = AsyncMock(side_effect=request_side_effect)  # type: ignore[attr-defined]
        cache = EntityMetadataCache(client, ttl=3600.0)  # type: ignore[arg-type]
        return cache, client

    async def test_registry_entity_known_without_state_lookup(self):
        cache, client = self._cache()
        assert await cache.is_known_batch(["light.a"]) == {"light.a": True}
        client._request.assert_not_awaited()

    async def test_state_only_entity_known(self):
        cache, client = self._cache()
        assert await cache.is_known_batch(["group.legacy"]) == {"group.legacy": True}
        client._request.assert_awaited_once_with("GET", "/states/group.legacy")

    async def test_404_is_unknown(self):
        from ha_mcp.client.rest_client import HomeAssistantAPIError

        cache, _ = self._cache(HomeAssistantAPIError("nope", status_code=404))
        assert await cache.is_known_batch(["spotify.track"]) == {"spotify.track": False}

    async def test_lookup_error_is_undetermined(self):
        from ha_mcp.client.rest_client import HomeAssistantConnectionError

        cache, _ = self._cache(HomeAssistantConnectionError("down"))
        assert await cache.is_known_batch(["light.b"]) == {"light.b": None}

    @pytest.mark.parametrize("value", ["light", "light.a/../x", "LIGHT.A", ""])
    async def test_not_entity_shaped_is_unknown_without_lookup(self, value: str):
        cache, client = self._cache()
        assert await cache.is_known_batch([value]) == {value: False}
        client._request.assert_not_awaited()

    async def test_misses_are_looked_up_concurrently(self):
        import asyncio

        in_flight: set[str] = set()
        peak = 0

        async def _request(_method: str, endpoint: str) -> dict[str, Any]:
            nonlocal peak
            in_flight.add(endpoint)
            peak = max(peak, len(in_flight))
            await asyncio.sleep(0)
            in_flight.discard(endpoint)
            return {}

        cache, _ = self._cache(_request)
        known = await cache.is_known_batch(["light.a", "group.x", "group.y", "bad"])
        assert known == {"light.a": True, "group.x": True, "group.y": True, "bad": False}
        assert peak == 2


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
