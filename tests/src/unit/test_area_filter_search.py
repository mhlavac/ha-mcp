"""Unit tests for area-filtered entity search (issue #504).

Tests that get_entities_by_area uses HA registries for accurate area resolution
instead of fuzzy-matching on friendly names, which caused false positives.

Bug: ha_search_entities(area_filter="salon") returned entities from unrelated
areas because the old implementation used fuzzy matching on entity friendly
names instead of consulting the entity/device/area registries.
"""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ha_mcp.tools.smart_search import SmartSearchTools, match_area_ids
from ha_mcp.tools.tools_search import register_search_tools


def _ws_success(result):
    """Wrap a result list in a successful WebSocket response."""
    return {"success": True, "result": result}


class MockClientWithRegistries:
    """Mock Home Assistant client that returns configurable registry data."""

    def __init__(
        self,
        entities: list[dict],
        areas: list[dict] | None = None,
        entity_registry: list[dict] | None = None,
        device_registry: list[dict] | None = None,
    ):
        self.entities = entities
        self.areas = areas or []
        self.entity_registry = entity_registry or []
        self.device_registry = device_registry or []
        self.base_url = "http://localhost:8123"

    async def get_states(self) -> list[dict]:
        return self.entities

    async def get_config(self) -> dict:
        return {"version": "2026.1.0"}

    async def send_websocket_message(self, message: dict) -> dict:
        msg_type = message.get("type", "")
        if msg_type == "config/area_registry/list":
            return _ws_success(self.areas)
        elif msg_type == "config/entity_registry/list":
            return _ws_success(self.entity_registry)
        elif msg_type == "config/device_registry/list":
            return _ws_success(self.device_registry)
        return {"success": False, "error": {"code": "unknown_command"}}


@pytest.fixture
def two_area_setup():
    """Setup with two areas: 'salon' and 'abc', each with distinct entities.

    This reproduces the exact scenario from issue #504:
    - Area 'salon' has a light entity
    - Area 'abc' has a climate entity (assigned via device)
    - Searching for area 'salon' must NOT return climate.abc
    """
    areas = [
        {"area_id": "salon", "name": "Salon"},
        {"area_id": "abc", "name": "ABC"},
    ]

    entities = [
        {
            "entity_id": "light.salon_lamp",
            "attributes": {"friendly_name": "Salon Lamp"},
            "state": "on",
        },
        {
            "entity_id": "climate.abc",
            "attributes": {"friendly_name": "Climatisation ABC"},
            "state": "heat",
        },
        {
            "entity_id": "sensor.outdoor_temp",
            "attributes": {"friendly_name": "Outdoor Temperature"},
            "state": "15.2",
        },
    ]

    entity_registry = [
        # light.salon_lamp directly assigned to salon area
        {"entity_id": "light.salon_lamp", "area_id": "salon", "device_id": None},
        # climate.abc assigned to area via device (no direct area_id)
        {"entity_id": "climate.abc", "area_id": None, "device_id": "device_abc_thermostat"},
        # sensor has no area at all
        {"entity_id": "sensor.outdoor_temp", "area_id": None, "device_id": None},
    ]

    device_registry = [
        {"id": "device_abc_thermostat", "area_id": "abc"},
    ]

    return {
        "areas": areas,
        "entities": entities,
        "entity_registry": entity_registry,
        "device_registry": device_registry,
    }


class TestAreaFilterAccuracy:
    """Test that area filtering uses registries and returns only correct entities."""

    @pytest.mark.asyncio
    async def test_area_filter_excludes_other_areas(self, two_area_setup):
        """Issue #504: searching for 'salon' must NOT return entities from area 'abc'."""
        client = MockClientWithRegistries(
            entities=two_area_setup["entities"],
            areas=two_area_setup["areas"],
            entity_registry=two_area_setup["entity_registry"],
            device_registry=two_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("salon")

        assert result["total_areas_found"] == 1
        assert result["total_entities"] == 1

        # Should only contain salon area
        assert "salon" in result["areas"]
        assert "abc" not in result["areas"]

        # Should only contain the salon lamp, not climate.abc
        salon_area = result["areas"]["salon"]
        all_entity_ids = []
        for domain_entities in salon_area["entities"].values():
            all_entity_ids.extend([e["entity_id"] for e in domain_entities])

        assert "light.salon_lamp" in all_entity_ids
        assert "climate.abc" not in all_entity_ids

    @pytest.mark.asyncio
    async def test_area_filter_returns_abc_area(self, two_area_setup):
        """Searching for 'abc' returns only entities in area 'abc'."""
        client = MockClientWithRegistries(
            entities=two_area_setup["entities"],
            areas=two_area_setup["areas"],
            entity_registry=two_area_setup["entity_registry"],
            device_registry=two_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("abc")

        assert result["total_areas_found"] == 1
        assert "abc" in result["areas"]
        assert "salon" not in result["areas"]

        abc_area = result["areas"]["abc"]
        all_entity_ids = []
        for domain_entities in abc_area["entities"].values():
            all_entity_ids.extend([e["entity_id"] for e in domain_entities])

        assert "climate.abc" in all_entity_ids
        assert "light.salon_lamp" not in all_entity_ids

    @pytest.mark.asyncio
    async def test_device_area_inheritance(self, two_area_setup):
        """Entities inherit area from their device when no direct area assignment."""
        client = MockClientWithRegistries(
            entities=two_area_setup["entities"],
            areas=two_area_setup["areas"],
            entity_registry=two_area_setup["entity_registry"],
            device_registry=two_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("abc")

        # climate.abc should be found via device_abc_thermostat -> area "abc"
        assert result["total_entities"] == 1
        abc_area = result["areas"]["abc"]
        all_entity_ids = []
        for domain_entities in abc_area["entities"].values():
            all_entity_ids.extend([e["entity_id"] for e in domain_entities])
        assert "climate.abc" in all_entity_ids

    @pytest.mark.asyncio
    async def test_unassigned_entities_excluded(self, two_area_setup):
        """Entities without any area assignment are not returned."""
        client = MockClientWithRegistries(
            entities=two_area_setup["entities"],
            areas=two_area_setup["areas"],
            entity_registry=two_area_setup["entity_registry"],
            device_registry=two_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        # Check salon
        salon_result = await tools.get_entities_by_area("salon")
        salon_ids = []
        for domain_entities in salon_result["areas"]["salon"]["entities"].values():
            salon_ids.extend([e["entity_id"] for e in domain_entities])
        assert "sensor.outdoor_temp" not in salon_ids

        # Check abc
        abc_result = await tools.get_entities_by_area("abc")
        abc_ids = []
        for domain_entities in abc_result["areas"]["abc"]["entities"].values():
            abc_ids.extend([e["entity_id"] for e in domain_entities])
        assert "sensor.outdoor_temp" not in abc_ids


class TestAreaMatchingLogic:
    """Test fuzzy matching on area names/IDs."""

    @pytest.fixture
    def multi_area_setup(self):
        areas = [
            {"area_id": "living_room", "name": "Living Room"},
            {"area_id": "bedroom_main", "name": "Main Bedroom"},
            {"area_id": "kitchen", "name": "Kitchen"},
        ]
        entities = [
            {
                "entity_id": "light.living_light",
                "attributes": {"friendly_name": "Living Room Light"},
                "state": "on",
            },
            {
                "entity_id": "light.bedroom_light",
                "attributes": {"friendly_name": "Main Bedroom Light"},
                "state": "off",
            },
            {
                "entity_id": "light.kitchen_light",
                "attributes": {"friendly_name": "Kitchen Light"},
                "state": "on",
            },
        ]
        entity_registry = [
            {"entity_id": "light.living_light", "area_id": "living_room", "device_id": None},
            {"entity_id": "light.bedroom_light", "area_id": "bedroom_main", "device_id": None},
            {"entity_id": "light.kitchen_light", "area_id": "kitchen", "device_id": None},
        ]
        return {
            "areas": areas,
            "entities": entities,
            "entity_registry": entity_registry,
            "device_registry": [],
        }

    @pytest.mark.asyncio
    async def test_exact_area_name_match(self, multi_area_setup):
        """Exact area name match works."""
        client = MockClientWithRegistries(
            entities=multi_area_setup["entities"],
            areas=multi_area_setup["areas"],
            entity_registry=multi_area_setup["entity_registry"],
            device_registry=multi_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("Kitchen")

        assert result["total_areas_found"] == 1
        assert "kitchen" in result["areas"]

    @pytest.mark.asyncio
    async def test_case_insensitive_area_match(self, multi_area_setup):
        """Area matching is case-insensitive."""
        client = MockClientWithRegistries(
            entities=multi_area_setup["entities"],
            areas=multi_area_setup["areas"],
            entity_registry=multi_area_setup["entity_registry"],
            device_registry=multi_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("kitchen")

        assert result["total_areas_found"] == 1
        assert "kitchen" in result["areas"]

    @pytest.mark.asyncio
    async def test_no_matching_area_returns_empty_with_suggestions(self, multi_area_setup):
        """Non-matching area returns empty results with available areas."""
        client = MockClientWithRegistries(
            entities=multi_area_setup["entities"],
            areas=multi_area_setup["areas"],
            entity_registry=multi_area_setup["entity_registry"],
            device_registry=multi_area_setup["device_registry"],
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("garage")

        assert result["total_areas_found"] == 0
        assert result["total_entities"] == 0
        assert result["areas"] == {}
        # Should include available areas
        available = result["available_areas"]
        area_ids = [a["area_id"] for a in available]
        assert "living_room" in area_ids
        assert "kitchen" in area_ids


class TestAreaFilterGroupByDomain:
    """Test domain grouping in area results."""

    @pytest.mark.asyncio
    async def test_grouped_by_domain(self):
        """Results are correctly grouped by domain when requested."""
        areas = [{"area_id": "salon", "name": "Salon"}]
        entities = [
            {"entity_id": "light.salon_light", "attributes": {"friendly_name": "Salon Light"}, "state": "on"},
            {"entity_id": "switch.salon_switch", "attributes": {"friendly_name": "Salon Switch"}, "state": "off"},
            {"entity_id": "sensor.salon_temp", "attributes": {"friendly_name": "Salon Temperature"}, "state": "21"},
        ]
        entity_registry = [
            {"entity_id": "light.salon_light", "area_id": "salon", "device_id": None},
            {"entity_id": "switch.salon_switch", "area_id": "salon", "device_id": None},
            {"entity_id": "sensor.salon_temp", "area_id": "salon", "device_id": None},
        ]
        client = MockClientWithRegistries(
            entities=entities,
            areas=areas,
            entity_registry=entity_registry,
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("salon", group_by_domain=True)

        salon_entities = result["areas"]["salon"]["entities"]
        assert isinstance(salon_entities, dict)  # grouped by domain
        assert "light" in salon_entities
        assert "switch" in salon_entities
        assert "sensor" in salon_entities
        assert len(salon_entities["light"]) == 1
        assert salon_entities["light"][0]["entity_id"] == "light.salon_light"

    @pytest.mark.asyncio
    async def test_flat_list_without_grouping(self):
        """Results are a flat list when group_by_domain=False."""
        areas = [{"area_id": "salon", "name": "Salon"}]
        entities = [
            {"entity_id": "light.salon_light", "attributes": {"friendly_name": "Salon Light"}, "state": "on"},
            {"entity_id": "switch.salon_switch", "attributes": {"friendly_name": "Salon Switch"}, "state": "off"},
        ]
        entity_registry = [
            {"entity_id": "light.salon_light", "area_id": "salon", "device_id": None},
            {"entity_id": "switch.salon_switch", "area_id": "salon", "device_id": None},
        ]
        client = MockClientWithRegistries(
            entities=entities,
            areas=areas,
            entity_registry=entity_registry,
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("salon", group_by_domain=False)

        salon_entities = result["areas"]["salon"]["entities"]
        assert isinstance(salon_entities, list)  # flat list
        entity_ids = [e["entity_id"] for e in salon_entities]
        assert "light.salon_light" in entity_ids
        assert "switch.salon_switch" in entity_ids


class TestEntityDirectAreaOverride:
    """Test that entity-level area_id takes priority over device-level area_id."""

    @pytest.mark.asyncio
    async def test_entity_area_overrides_device_area(self):
        """Entity's direct area assignment takes priority over device area."""
        areas = [
            {"area_id": "salon", "name": "Salon"},
            {"area_id": "bedroom", "name": "Bedroom"},
        ]
        entities = [
            {"entity_id": "sensor.device_temp", "attributes": {"friendly_name": "Device Temp"}, "state": "22"},
        ]
        # Entity is directly assigned to salon, but device is in bedroom
        entity_registry = [
            {"entity_id": "sensor.device_temp", "area_id": "salon", "device_id": "device_1"},
        ]
        device_registry = [
            {"id": "device_1", "area_id": "bedroom"},
        ]
        client = MockClientWithRegistries(
            entities=entities,
            areas=areas,
            entity_registry=entity_registry,
            device_registry=device_registry,
        )
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        # Should be in salon (entity override), not bedroom (device)
        salon_result = await tools.get_entities_by_area("salon")
        assert salon_result["total_entities"] == 1

        bedroom_result = await tools.get_entities_by_area("bedroom")
        assert bedroom_result["total_entities"] == 0


class TestRegistryFailureGracefulDegradation:
    """Test behavior when registry calls fail."""

    @pytest.mark.asyncio
    async def test_all_registries_fail_returns_empty(self):
        """When all registry calls fail, returns empty results gracefully."""

        class FailingClient:
            base_url = "http://localhost:8123"

            async def get_states(self):
                return [
                    {"entity_id": "light.test", "attributes": {"friendly_name": "Test"}, "state": "on"},
                ]

            async def get_config(self):
                return {}

            async def send_websocket_message(self, message):
                raise ConnectionError("WebSocket unavailable")

        client = FailingClient()
        tools = SmartSearchTools(client=client, fuzzy_threshold=60)

        result = await tools.get_entities_by_area("salon")

        # Should return empty, not crash
        assert result["total_areas_found"] == 0
        assert result["total_entities"] == 0


# ---------------------------------------------------------------------------
# Exact-first area matching, multi-area merge, area tags on results
# ---------------------------------------------------------------------------

# Two areas that both exactly match "emily" (one by id, one by name) plus a
# fuzzy cousin that must NOT be picked when exact matches exist.
_EMILY_AREAS = [
    {"area_id": "emily", "name": "Emily L1", "aliases": []},
    {"area_id": "emily_2", "name": "Emily", "aliases": ["Em's room"]},
    {"area_id": "emilys_den", "name": "Emilys Den"},
    {"area_id": "kitchen", "name": "Kitchen", "aliases": ["Cuisine"]},
]
_EMILY_STATES = [
    {"entity_id": eid, "attributes": {"friendly_name": name}, "state": "on"}
    for eid, name in [
        ("light.emily_ceiling", "Emily Ceiling"),
        ("switch.emily_fan", "Emily Fan"),
        ("light.emily_2_lamp", "Emily Lamp"),
        ("sensor.emily_2_temp", "Emily Temperature"),
        ("light.den", "Den Light"),
        ("light.kitchen", "Kitchen Light"),
        ("light.hallway", "Hallway Light"),
        ("light.desk", "Desk Light"),
    ]
]
_EMILY_ENTITY_REGISTRY = [
    {"entity_id": "light.emily_ceiling", "area_id": "emily", "device_id": None},
    {"entity_id": "switch.emily_fan", "area_id": "emily", "device_id": None},
    # area via device
    {"entity_id": "light.emily_2_lamp", "area_id": None, "device_id": "dev_e2"},
    {"entity_id": "sensor.emily_2_temp", "area_id": "emily_2", "device_id": None},
    {"entity_id": "light.den", "area_id": "emilys_den", "device_id": None},
    # entity area overrides its device's area
    {"entity_id": "light.kitchen", "area_id": "kitchen", "device_id": "dev_e2"},
    {"entity_id": "light.hallway", "area_id": None, "device_id": None},
    # light.desk: state only, no registry entry → no area
]
_EMILY_DEVICES = [{"id": "dev_e2", "area_id": "emily_2"}]


def _emily_client() -> MockClientWithRegistries:
    return MockClientWithRegistries(
        entities=_EMILY_STATES,
        areas=_EMILY_AREAS,
        entity_registry=_EMILY_ENTITY_REGISTRY,
        device_registry=_EMILY_DEVICES,
    )


def _search_tool(client: Any) -> Any:
    """Register the search tools against ``client`` and return ha_search_entities."""
    registered: dict[str, Any] = {}

    def tool(*_args: Any, **_kwargs: Any) -> Any:
        def wrapper(func: Any) -> Any:
            registered[func.__name__] = func
            return func

        return wrapper

    mcp = MagicMock()
    mcp.tool = tool
    smart_tools = SmartSearchTools(client=client, fuzzy_threshold=60)
    register_search_tools(mcp, client, smart_tools=smart_tools)
    return registered["ha_search_entities"]


async def _search(client: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("query", "")
    result = await _search_tool(client)(**kwargs)
    return result["data"]


class TestExactFirstAreaMatching:
    """B1: exact id/name/alias matches win; fuzzy only when none is exact."""

    _REGISTRY: ClassVar[dict[str, dict[str, Any]]] = {a["area_id"]: a for a in _EMILY_AREAS}

    def test_all_exact_matches_and_only_those(self):
        assert match_area_ids("emily", self._REGISTRY) == (["emily", "emily_2"], "exact")

    def test_exact_match_is_case_insensitive_and_trimmed(self):
        assert match_area_ids("  EMILY ", self._REGISTRY) == (
            ["emily", "emily_2"], "exact",
        )

    def test_exact_alias_match(self):
        assert match_area_ids("cuisine", self._REGISTRY) == (["kitchen"], "exact")
        assert match_area_ids("em's room", self._REGISTRY) == (["emily_2"], "exact")

    def test_exact_name_match_beats_fuzzy(self):
        assert match_area_ids("emily l1", self._REGISTRY) == (["emily"], "exact")

    def test_fuzzy_fallback_when_nothing_exact(self):
        assert match_area_ids("kitchn", self._REGISTRY) == (["kitchen"], "fuzzy")

    def test_no_match(self):
        assert match_area_ids("garage", self._REGISTRY) == ([], None)

    async def test_get_entities_by_area_returns_every_exact_area(self):
        tools = SmartSearchTools(client=_emily_client(), fuzzy_threshold=60)
        result = await tools.get_entities_by_area("emily")
        assert list(result["areas"]) == ["emily", "emily_2"]
        assert result["area_match"] == "exact"
        assert result["total_entities"] == 4


class TestAreaOnlySearch:
    """B2: area_filter + empty query merges every matched area, paginated."""

    async def test_merges_all_matched_areas_with_area_tags(self):
        data = await _search(_emily_client(), area_filter="emily", limit=50)
        assert data["search_type"] == "area_only"
        assert {(r["entity_id"], r["area_id"], r["area_name"]) for r in data["results"]} == {
            ("light.emily_ceiling", "emily", "Emily L1"),
            ("switch.emily_fan", "emily", "Emily L1"),
            ("light.emily_2_lamp", "emily_2", "Emily"),
            ("sensor.emily_2_temp", "emily_2", "Emily"),
        }
        assert data["total_matches"] == 4
        assert data["area_name"] == "Emily L1, Emily"
        assert data["matched_areas"] == [
            {"area_id": "emily", "area_name": "Emily L1", "entity_count": 2},
            {"area_id": "emily_2", "area_name": "Emily", "entity_count": 2},
        ]

    async def test_domain_filter_honoured(self):
        data = await _search(
            _emily_client(), area_filter="emily", domain_filter="light", limit=50
        )
        assert {r["entity_id"] for r in data["results"]} == {
            "light.emily_ceiling", "light.emily_2_lamp",
        }
        assert data["total_matches"] == 2
        assert data["domain_filter"] == "light"
        assert set(data["by_domain"]) == {"light"}
        assert [a["entity_count"] for a in data["matched_areas"]] == [1, 1]

    async def test_paginated_like_domain_listing(self):
        client = _emily_client()
        page1 = await _search(client, area_filter="emily", limit=3)
        page2 = await _search(client, area_filter="emily", limit=3, offset=3)
        assert (page1["total_matches"], page1["count"], page1["has_more"]) == (4, 3, True)
        assert page1["next_offset"] == 3
        assert (page2["count"], page2["has_more"], page2["next_offset"]) == (1, False, None)
        ids1 = [r["entity_id"] for r in page1["results"]]
        ids2 = [r["entity_id"] for r in page2["results"]]
        assert len(set(ids1 + ids2)) == 4
        # by_domain groups exactly the returned page
        for page in (page1, page2):
            grouped = [e["entity_id"] for ents in page["by_domain"].values() for e in ents]
            assert sorted(grouped) == sorted(r["entity_id"] for r in page["results"])

    async def test_no_matching_area(self):
        data = await _search(_emily_client(), area_filter="garage")
        assert data["success"] is True
        assert data["total_matches"] == 0
        assert data["results"] == []
        assert data["matched_areas"] == []
        assert "garage" in data["message"]
        assert {a["area_id"] for a in data["available_areas"]} >= {"emily", "kitchen"}

    async def test_area_with_query_also_tagged_and_domain_filtered(self):
        data = await _search(
            _emily_client(), query="emily", area_filter="emily",
            domain_filter="light", limit=50, exact_match=False,
        )
        assert data["search_type"] == "area_filtered_query"
        assert data["results"], data
        for r in data["results"]:
            assert r["domain"] == "light"
            assert r["area_id"] in {"emily", "emily_2"}
            assert r["area_name"] in {"Emily L1", "Emily"}


class TestDomainListingAreas:
    """B3: the domain listing tags each result with its area."""

    async def test_domain_listing_carries_area(self):
        data = await _search(_emily_client(), domain_filter="light", limit=50)
        assert data["search_type"] == "domain_listing"
        areas = {r["entity_id"]: (r["area_id"], r["area_name"]) for r in data["results"]}
        assert areas == {
            "light.emily_ceiling": ("emily", "Emily L1"),
            "light.emily_2_lamp": ("emily_2", "Emily"),  # via its device
            "light.den": ("emilys_den", "Emilys Den"),
            "light.kitchen": ("kitchen", "Kitchen"),  # own area beats device's
            "light.hallway": (None, None),
            "light.desk": (None, None),  # no registry entry
        }

    async def test_domain_listing_survives_registry_failure(self):
        client = _emily_client()
        client.send_websocket_message = AsyncMock(  # type: ignore[method-assign]
            return_value={"success": False, "error": "boom"}
        )
        data = await _search(client, domain_filter="light", limit=50)
        assert data["count"] == 6
        assert all(r["area_id"] is None and r["area_name"] is None for r in data["results"])


class TestAreaSearchUnderPolicy:
    """An unreadable entity in a matched area never appears in any result."""

    @staticmethod
    def _policy_client() -> Any:
        from ha_mcp.access_policy import (
            AccessPolicy,
            EntitiesPolicy,
            EntityMeta,
            EntityMetadataCache,
            EntityRule,
            PolicyConfig,
        )
        from ha_mcp.client.rest_client import HomeAssistantClient

        hidden = {"switch.emily_fan", "sensor.emily_2_temp"}
        cache = EntityMetadataCache(client=None, ttl=60.0)  # type: ignore[arg-type]
        cache._entities = {
            e["entity_id"]: EntityMeta(
                entity_id=e["entity_id"],
                area_id=e["area_id"] or ("emily_2" if e["device_id"] else None),
                device_id=e["device_id"],
            )
            for e in _EMILY_ENTITY_REGISTRY
        }
        cache._area_labels = {}
        cache._fetched_at = float("inf")  # never refresh
        policy = AccessPolicy(
            PolicyConfig(
                entities=EntitiesPolicy(
                    allow=[EntityRule(entity_globs=["light.*", "switch.*", "sensor.*"])],
                    deny=[EntityRule(entity_globs=sorted(hidden))],
                )
            ),
            cache,
        )

        with patch.object(HomeAssistantClient, "__init__", lambda self, **kw: None):
            client = HomeAssistantClient()
        client.base_url = "http://test.local:8123"
        client.token = "test-token"
        client.timeout = 30
        client.httpx_client = MagicMock()
        client.policy = policy

        async def _request(method: str, endpoint: str, **kwargs: Any) -> Any:
            if endpoint == "/states":
                return _EMILY_STATES
            if endpoint == "/config":
                return {"time_zone": "UTC"}
            raise AssertionError(f"unexpected request {method} {endpoint}")

        client._request = _request  # type: ignore[method-assign]
        registries = {
            "config/area_registry/list": _EMILY_AREAS,
            "config/entity_registry/list": _EMILY_ENTITY_REGISTRY,
            "config/device_registry/list": _EMILY_DEVICES,
        }
        ws = MagicMock()
        ws.send_command = AsyncMock(
            side_effect=lambda command, **_kw: {
                "success": True, "result": [dict(e) for e in registries[command]],
            }
        )
        return client, ws, hidden

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"area_filter": "emily"},
            {"area_filter": "emily", "domain_filter": "switch"},
            {"area_filter": "emily", "query": "fan", "exact_match": False},
            {"area_filter": "emily", "query": "temperature", "exact_match": False},
            {"domain_filter": "switch"},
            {"domain_filter": "sensor"},
        ],
    )
    async def test_unreadable_entity_never_listed(self, kwargs: dict[str, Any]):
        client, ws, hidden = self._policy_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ):
            data = await _search(client, limit=50, **kwargs)
        assert not any(h in str(data) for h in hidden), data

    async def test_readable_entities_of_both_areas_still_listed(self):
        client, ws, _ = self._policy_client()
        with patch(
            "ha_mcp.client.websocket_client.get_websocket_client",
            AsyncMock(return_value=ws),
        ):
            data = await _search(client, area_filter="emily", limit=50)
        assert {r["entity_id"] for r in data["results"]} == {
            "light.emily_ceiling", "light.emily_2_lamp",
        }
        assert [a["entity_count"] for a in data["matched_areas"]] == [1, 1]
