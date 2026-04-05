"""
Fine-grained access control policy for Home Assistant MCP Server.

Policies are loaded from a YAML file referenced by the HAMCP_POLICY_FILE env var
and gate both tool registration (by name / category tag) and entity operations
(by area, HA label, entity_id glob, device_id).

Design goals:
- Deny-by-default when a policy is loaded (principle of least privilege).
- Unload the policy entirely when HAMCP_POLICY_FILE is unset — backward compat.
- Hide entities outside scope from search/list/state (invisible mode), not just
  block writes.
- Use HA's own entity/device/area/label registries as the source of truth for
  matching — no duplicated metadata.

See docs/access-policy.md for the YAML schema and example policies.
"""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from .client.rest_client import HomeAssistantClient

logger = logging.getLogger(__name__)


# When True, ``HomeAssistantClient.send_websocket_message`` skips the
# post-response policy filter. Set while the policy's own metadata cache is
# populating itself — the cache uses ``send_websocket_message`` to fetch
# ``entity_registry/list`` + ``device_registry/list``, and running the filter
# on those responses both corrupts the cache (empty result) and recurses
# into the cache via ``can_read_batch`` while the cache's refresh lock is
# still held — a deadlock.
#
# Lives in access_policy.py (not rest_client.py) because the cache owns this
# concern: only cache-originated WS calls should bypass filtering, and the
# cache is the authoritative place that decides when to enter that state.
bypass_policy_filter: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ha_mcp_bypass_policy_filter", default=False
)


# ---------------------------------------------------------------------------
# Policy schema (Pydantic)
# ---------------------------------------------------------------------------


class EntityRule(BaseModel):
    """One matcher clause. A rule matches an entity if ANY of its primitives match (OR).

    Primitives:
      - areas:        entity's resolved area_id is in this list
      - labels:       entity's resolved labels intersect this list
      - entity_globs: entity_id matches one of these fnmatch patterns
      - device_ids:   entity's device_id is in this list

    OR semantics within a rule means these primitives are alternative ways to
    identify the same set of entities. If you want AND semantics (e.g. "in
    area X AND labeled Y"), use separate rules and/or the allow/deny interplay.

    A rule with NO primitives matches nothing. Empty-primitive lists are ignored.
    """

    model_config = ConfigDict(extra="forbid")

    areas: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    entity_globs: list[str] = Field(default_factory=list)
    device_ids: list[str] = Field(default_factory=list)


class EntitiesPolicy(BaseModel):
    """Entity-level access rules.

    Precedence: deny > allow > default_action.
    Write check: can_read(e) AND NOT write_deny AND (write_allow is None OR write_allow matches).
    """

    model_config = ConfigDict(extra="forbid")

    allow: list[EntityRule] = Field(default_factory=list)
    deny: list[EntityRule] = Field(default_factory=list)
    # write_allow: when None (unset), inherits from `allow`. When set to an
    # empty list, no entity is writable (pure read-only view). When set to a
    # non-empty list, only entities matching these rules are writable.
    write_allow: list[EntityRule] | None = None
    write_deny: list[EntityRule] = Field(default_factory=list)


class ToolsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disabled_names: list[str] = Field(default_factory=list)
    disabled_tags: list[str] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    """Top-level policy schema mirroring the YAML file."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    # Applied when no rule (allow/deny) matches. Defaults to `deny` — principle
    # of least privilege. Set to `allow` for a permissive policy with carve-outs.
    default_action: Literal["allow", "deny"] = "deny"
    # When True, all write operations are blocked regardless of entity rules.
    readonly_mode: bool = False
    tools: ToolsPolicy = Field(default_factory=ToolsPolicy)
    entities: EntitiesPolicy = Field(default_factory=EntitiesPolicy)

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                f"Policy version {v} is not supported by this server (expected 1). "
                "Upgrade ha-mcp or downgrade the policy file."
            )
        return v


# ---------------------------------------------------------------------------
# Entity metadata cache
# ---------------------------------------------------------------------------


@dataclass
class EntityMeta:
    """Resolved metadata for a single entity.

    `area_id` and `labels` already include the entity→device fallback, matching
    HA's own resolution model:
      - area_id = entity.area_id OR device.area_id
      - labels  = union(entity.labels, device.labels)

    `labels` contains BOTH the label IDs (slugs, as HA stores them on entities)
    AND their resolved human names (from the label registry). A rule like
    ``labels: ["owner:alice"]`` matches a label with ID ``owner_alice`` and
    name ``owner:alice`` — whichever spelling the user writes.
    """

    entity_id: str
    area_id: str | None = None
    labels: set[str] = field(default_factory=set)
    device_id: str | None = None


class EntityMetadataCache:
    """TTL-refreshed cache of entity metadata for policy evaluation.

    Fetches entity_registry + device_registry in parallel (pattern from
    smart_search.py:208-255) and precomputes entity→device fallback
    (smart_search.py:291-300).

    Thread-safety: refreshes are serialized via an asyncio.Lock so concurrent
    callers wait for a single in-flight refresh rather than stampeding the
    registries.
    """

    DEFAULT_TTL = 60.0  # seconds

    def __init__(self, client: HomeAssistantClient, ttl: float = DEFAULT_TTL):
        self._client = client
        self._ttl = ttl
        self._entities: dict[str, EntityMeta] = {}
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    async def ensure_fresh(self) -> None:
        """Refresh cache if TTL has expired. Safe under concurrent callers."""
        if time.monotonic() - self._fetched_at < self._ttl:
            return
        async with self._lock:
            # Re-check after acquiring the lock in case another coroutine refreshed.
            if time.monotonic() - self._fetched_at < self._ttl:
                return
            await self._refresh()

    async def _refresh(self) -> None:
        """Fetch registries and rebuild the entity metadata map.

        Bypasses policy filtering on its own WS calls — filtering those
        responses would both corrupt the cache (empty result) and recurse
        into ``can_read_batch`` while the refresh lock is still held
        (deadlock).
        """
        bypass_token = bypass_policy_filter.set(True)
        try:
            entity_reg_task = self._client.send_websocket_message(
                {"type": "config/entity_registry/list"}
            )
            device_reg_task = self._client.send_websocket_message(
                {"type": "config/device_registry/list"}
            )
            label_reg_task = self._client.send_websocket_message(
                {"type": "config/label_registry/list"}
            )
            results = await asyncio.gather(
                entity_reg_task,
                device_reg_task,
                label_reg_task,
                return_exceptions=True,
            )
        finally:
            bypass_policy_filter.reset(bypass_token)

        # label_id -> name. HA stores label IDs (slugs like "owner_alice") on
        # entities/devices, but users naturally write policy rules using label
        # NAMES ("owner:alice"). We expand meta.labels to include both IDs and
        # names so rules can reference labels by either spelling.
        label_id_to_name: dict[str, str] = {}
        if isinstance(results[2], dict) and results[2].get("success"):
            for lbl in results[2].get("result", []):
                lid = lbl.get("label_id")
                lname = lbl.get("name")
                if lid and lname:
                    label_id_to_name[lid] = lname
        elif isinstance(results[2], Exception):
            logger.warning("label_registry fetch failed: %s", results[2])

        def _expand_labels(label_ids: Iterable[str]) -> set[str]:
            """Return the union of label IDs and their resolved names."""
            ids = set(label_ids)
            names = {label_id_to_name[lid] for lid in ids if lid in label_id_to_name}
            return ids | names

        # device_id -> (area_id, labels set — with names expanded)
        device_info: dict[str, tuple[str | None, set[str]]] = {}
        if isinstance(results[1], dict) and results[1].get("success"):
            for device in results[1].get("result", []):
                dev_id = device.get("id")
                if dev_id:
                    device_info[dev_id] = (
                        device.get("area_id"),
                        _expand_labels(device.get("labels") or []),
                    )
        elif isinstance(results[1], Exception):
            logger.warning("device_registry fetch failed: %s", results[1])

        entities: dict[str, EntityMeta] = {}
        if isinstance(results[0], dict) and results[0].get("success"):
            for entry in results[0].get("result", []):
                entity_id = entry.get("entity_id")
                if not entity_id:
                    continue
                entity_area = entry.get("area_id")
                entity_labels = _expand_labels(entry.get("labels") or [])
                device_id = entry.get("device_id")

                # Fallback to device metadata
                dev_area, dev_labels = device_info.get(device_id, (None, set()))
                resolved_area = entity_area or dev_area
                resolved_labels = entity_labels | dev_labels

                entities[entity_id] = EntityMeta(
                    entity_id=entity_id,
                    area_id=resolved_area,
                    labels=resolved_labels,
                    device_id=device_id,
                )
        # Always update _fetched_at so a transient failure doesn't leave the
        # cache permanently "expired" (which would stampede every can_read call)
        # nor permanently "fresh" (which would hide newly-created entities).
        self._fetched_at = time.monotonic()

        if isinstance(results[0], Exception):
            logger.warning(
                "entity_registry fetch failed, keeping previous cache: %s",
                results[0],
            )
            return

        self._entities = entities
        logger.debug("AccessPolicy metadata cache refreshed: %d entities", len(entities))

    def _lookup(self, entity_id: str) -> EntityMeta:
        """In-memory lookup from the current snapshot (no freshness guarantee).

        Returns an empty EntityMeta for unknown entities so rule matching can
        run uniformly. Internal — AccessPolicy uses this after a single
        ensure_fresh() when evaluating many entities.
        """
        meta = self._entities.get(entity_id)
        if meta is None:
            return EntityMeta(entity_id=entity_id)
        return meta

    async def get(self, entity_id: str) -> EntityMeta:
        """Ensure cache is fresh, then return entity metadata."""
        await self.ensure_fresh()
        return self._lookup(entity_id)

    def invalidate(self) -> None:
        """Force a refresh on the next access (e.g. in response to a WS event)."""
        self._fetched_at = 0.0


# ---------------------------------------------------------------------------
# Access policy
# ---------------------------------------------------------------------------


# Under readonly_mode we block any tool that is not explicitly read-only.
# This is stricter than just "destructive" — creating a record (destructiveHint=False)
# is still a write and has side effects, so it is blocked too.
def _is_write_tool(annotations: dict[str, Any] | None) -> bool:
    ann = annotations or {}
    return ann.get("readOnlyHint") is not True


class AccessPolicy:
    """Policy engine.

    All methods require a metadata cache to evaluate entity-level rules.
    `is_tool_allowed` is synchronous (no cache lookup required).
    """

    def __init__(self, cfg: PolicyConfig, cache: EntityMetadataCache):
        self.cfg = cfg
        self.cache = cache
        self._disabled_names = set(cfg.tools.disabled_names)
        self._disabled_tags = set(cfg.tools.disabled_tags)

    # -- Tool gating --------------------------------------------------------

    def is_tool_allowed(
        self,
        name: str,
        tags: set[str] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> bool:
        if name in self._disabled_names:
            return False
        if tags and (tags & self._disabled_tags):
            return False
        return not (self.cfg.readonly_mode and _is_write_tool(annotations))

    # -- Entity gating ------------------------------------------------------

    async def can_read(self, entity_id: str) -> bool:
        meta = await self.cache.get(entity_id)
        return self._check_read(entity_id, meta)

    async def can_write(self, entity_id: str) -> bool:
        if self.cfg.readonly_mode:
            return False
        meta = await self.cache.get(entity_id)
        if not self._check_read(entity_id, meta):
            return False
        if self._matches_any(self.cfg.entities.write_deny, entity_id, meta):
            return False
        write_allow = self.cfg.entities.write_allow
        if write_allow is None:
            # Inherit from `allow` (already confirmed via can_read above)
            return True
        # Explicit write_allow list (may be empty → nothing writable)
        return self._matches_any(write_allow, entity_id, meta)

    async def filter_entities(self, entity_ids: Iterable[str]) -> list[str]:
        """Return only the entity_ids that are readable. Ensures single cache refresh."""
        await self.cache.ensure_fresh()
        return [
            eid for eid in entity_ids
            if self._check_read(eid, self.cache._lookup(eid))
        ]

    async def can_read_batch(self, entity_ids: Iterable[str]) -> dict[str, bool]:
        """Map each entity_id to a read decision. Single cache refresh for the batch."""
        await self.cache.ensure_fresh()
        return {
            eid: self._check_read(eid, self.cache._lookup(eid))
            for eid in entity_ids
        }

    async def expand_targets(
        self,
        device_ids: Iterable[str] | None = None,
        area_ids: Iterable[str] | None = None,
        label_ids: Iterable[str] | None = None,
    ) -> set[str]:
        """Resolve HA service-call targets (device/area/label) to concrete entity_ids.

        Returns the union of entities matching any provided target. Empty collection
        arguments are ignored. A fresh cache snapshot is ensured before the scan.
        """
        devs = set(device_ids or ())
        areas = set(area_ids or ())
        labels = set(label_ids or ())
        if not (devs or areas or labels):
            return set()
        await self.cache.ensure_fresh()
        result: set[str] = set()
        for meta in self.cache._entities.values():
            if (
                (devs and meta.device_id in devs)
                or (areas and meta.area_id in areas)
                or (labels and meta.labels & labels)
            ):
                result.add(meta.entity_id)
        return result

    def _check_read(self, entity_id: str, meta: EntityMeta) -> bool:
        """Core read-permission logic given pre-fetched metadata.

        Applies the precedence deny > allow > default_action.
        """
        if self._matches_any(self.cfg.entities.deny, entity_id, meta):
            return False
        if self._matches_any(self.cfg.entities.allow, entity_id, meta):
            return True
        return self.cfg.default_action == "allow"

    # -- Matching primitives ------------------------------------------------

    def _matches_any(
        self, rules: list[EntityRule], entity_id: str, meta: EntityMeta
    ) -> bool:
        return any(self._rule_matches(rule, entity_id, meta) for rule in rules)

    @staticmethod
    def _rule_matches(rule: EntityRule, entity_id: str, meta: EntityMeta) -> bool:
        # areas: match only when entity has a resolved area
        if rule.areas and meta.area_id and meta.area_id in rule.areas:
            return True
        # labels: intersection (entity has any label named in the rule)
        if rule.labels and meta.labels and meta.labels.intersection(rule.labels):
            return True
        # entity_globs: fnmatch against the entity_id
        for pattern in rule.entity_globs:
            if fnmatch.fnmatchcase(entity_id, pattern):
                return True
        # device_ids: direct membership
        return bool(
            rule.device_ids and meta.device_id and meta.device_id in rule.device_ids
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class PolicyLoadError(Exception):
    """Raised when a policy file cannot be loaded or validated."""


def load_policy_from_file(path: str | Path) -> PolicyConfig:
    """Load a policy YAML file. Raises PolicyLoadError on any failure.

    Missing file, malformed YAML, and schema validation errors are all fatal
    (fail-closed). Callers should surface the error to the user at startup.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise PolicyLoadError(f"Policy file not found: {resolved}")

    yaml = YAML(typ="safe")
    try:
        with resolved.open("r", encoding="utf-8") as f:
            raw = yaml.load(f)
    except YAMLError as e:
        raise PolicyLoadError(f"Policy file {resolved} is malformed YAML: {e}") from e

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise PolicyLoadError(
            f"Policy file {resolved} must contain a YAML mapping at the top level"
        )

    try:
        return PolicyConfig.model_validate(raw)
    except ValidationError as e:
        raise PolicyLoadError(f"Policy file {resolved} failed validation: {e}") from e


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_policy: AccessPolicy | None = None


def set_global_policy(policy: AccessPolicy | None) -> None:
    """Install (or clear) the process-wide policy instance."""
    global _policy
    _policy = policy


def get_global_policy() -> AccessPolicy | None:
    """Return the currently-installed policy, or None if unset."""
    return _policy
