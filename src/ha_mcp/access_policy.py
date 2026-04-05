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
from collections.abc import Callable, Iterable
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

    ``area_id`` and ``labels`` include HA's entity→device fallback:
    ``area_id = entity.area_id OR device.area_id``, and
    ``labels = union(entity.labels, device.labels)``. The ``labels`` set
    also includes resolved human names from the label registry (see
    EntityMetadataCache for details).
    """

    entity_id: str
    area_id: str | None = None
    labels: set[str] = field(default_factory=set)
    device_id: str | None = None


def _reduce_label_registry(result: Any) -> dict[str, str]:
    """Extract ``label_id -> name`` from a ``label_registry/list`` response."""
    if isinstance(result, Exception):
        logger.warning("label_registry fetch failed: %s", result)
        return {}
    if not (isinstance(result, dict) and result.get("success")):
        return {}
    out: dict[str, str] = {}
    for lbl in result.get("result", []):
        lid, lname = lbl.get("label_id"), lbl.get("name")
        if lid and lname:
            out[lid] = lname
    return out


def _label_expander(
    label_id_to_name: dict[str, str],
) -> Callable[[Iterable[str]], set[str]]:
    """Build a closure that expands label IDs to ``IDs | resolved names``."""
    def expand(label_ids: Iterable[str]) -> set[str]:
        ids = set(label_ids)
        return ids | {label_id_to_name[lid] for lid in ids if lid in label_id_to_name}
    return expand


def _reduce_device_registry(
    result: Any, expand_labels: Callable[[Iterable[str]], set[str]]
) -> dict[str, tuple[str | None, set[str]]]:
    """Extract ``device_id -> (area_id, expanded_labels)`` from the response."""
    if isinstance(result, Exception):
        logger.warning("device_registry fetch failed: %s", result)
        return {}
    if not (isinstance(result, dict) and result.get("success")):
        return {}
    out: dict[str, tuple[str | None, set[str]]] = {}
    for device in result.get("result", []):
        dev_id = device.get("id")
        if dev_id:
            out[dev_id] = (
                device.get("area_id"),
                expand_labels(device.get("labels") or []),
            )
    return out


def _reduce_entity_registry(
    result: Any,
    device_info: dict[str, tuple[str | None, set[str]]],
    expand_labels: Callable[[Iterable[str]], set[str]],
) -> tuple[dict[str, EntityMeta], bool]:
    """Build the entity metadata map, applying entity→device fallback.

    Returns ``(entities, ok)`` where ``ok`` is False on fetch failure so the
    caller can keep the previous cache snapshot.
    """
    if isinstance(result, Exception):
        return {}, False
    if not (isinstance(result, dict) and result.get("success")):
        return {}, True
    entities: dict[str, EntityMeta] = {}
    for entry in result.get("result", []):
        entity_id = entry.get("entity_id")
        if not entity_id:
            continue
        device_id = entry.get("device_id")
        dev_area, dev_labels = device_info.get(device_id, (None, set()))
        entities[entity_id] = EntityMeta(
            entity_id=entity_id,
            area_id=entry.get("area_id") or dev_area,
            labels=expand_labels(entry.get("labels") or []) | dev_labels,
            device_id=device_id,
        )
    return entities, True


class EntityMetadataCache:
    """TTL-refreshed cache of entity metadata for policy evaluation.

    Fetches entity/device/label registries in parallel and precomputes the
    entity→device fallback for ``area_id`` and ``labels``.

    Labels union: HA stores label IDs (slugs like ``owner_alice``) on
    entities/devices, but users naturally write rules using label NAMES
    (``"owner:alice"``). The cache expands each entity's label set to the
    union of IDs and their resolved names so rules match either spelling.

    Thread-safety: refreshes are serialized via an ``asyncio.Lock`` so
    concurrent callers wait for a single in-flight refresh rather than
    stampeding the registries.
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
        entity_result, device_result, label_result = await self._fetch_registries()

        label_id_to_name = _reduce_label_registry(label_result)
        expand = _label_expander(label_id_to_name)
        device_info = _reduce_device_registry(device_result, expand)
        entities, entity_ok = _reduce_entity_registry(entity_result, device_info, expand)

        # Always update _fetched_at so a transient failure doesn't leave the
        # cache permanently "expired" (stampede) nor permanently "fresh"
        # (hiding newly-created entities).
        self._fetched_at = time.monotonic()

        if not entity_ok:
            logger.warning(
                "entity_registry fetch failed, keeping previous cache: %s",
                entity_result,
            )
            return

        self._entities = entities
        logger.debug("AccessPolicy metadata cache refreshed: %d entities", len(entities))

    async def _fetch_registries(self) -> tuple[Any, Any, Any]:
        """Fetch entity/device/label registries in parallel with filter bypass."""
        bypass_token = bypass_policy_filter.set(True)
        try:
            results = await asyncio.gather(
                self._client.send_websocket_message(
                    {"type": "config/entity_registry/list"}
                ),
                self._client.send_websocket_message(
                    {"type": "config/device_registry/list"}
                ),
                self._client.send_websocket_message(
                    {"type": "config/label_registry/list"}
                ),
                return_exceptions=True,
            )
        finally:
            bypass_policy_filter.reset(bypass_token)
        entity_result, device_result, label_result = results
        return entity_result, device_result, label_result

    def iter_entities(self) -> Iterable[EntityMeta]:
        """Iterate the current snapshot of cached entity metadata.

        Stable during a single call because ``_refresh`` assigns a new dict
        atomically. Freshness is the caller's responsibility — call
        ``ensure_fresh()`` first if needed.
        """
        return self._entities.values()

    def entities_by_device(self) -> dict[str, list[str]]:
        """Group the current snapshot by ``device_id``.

        Entities without a device are omitted. No freshness guarantee —
        callers must ``ensure_fresh()`` first if they need current data.
        """
        grouped: dict[str, list[str]] = {}
        for meta in self._entities.values():
            if meta.device_id:
                grouped.setdefault(meta.device_id, []).append(meta.entity_id)
        return grouped

    def lookup(self, entity_id: str) -> EntityMeta:
        """In-memory lookup from the current snapshot (no freshness guarantee).

        Returns an empty ``EntityMeta`` for unknown entities so rule matching
        can run uniformly. Callers that need a fresh cache should call
        ``ensure_fresh()`` first (see ``get()`` for the combined form).
        """
        meta = self._entities.get(entity_id)
        if meta is None:
            return EntityMeta(entity_id=entity_id)
        return meta

    async def get(self, entity_id: str) -> EntityMeta:
        """Ensure cache is fresh, then return entity metadata."""
        await self.ensure_fresh()
        return self.lookup(entity_id)

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
            if self._check_read(eid, self.cache.lookup(eid))
        ]

    async def can_read_batch(self, entity_ids: Iterable[str]) -> dict[str, bool]:
        """Map each entity_id to a read decision. Single cache refresh for the batch."""
        await self.cache.ensure_fresh()
        return {
            eid: self._check_read(eid, self.cache.lookup(eid))
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
        for meta in self.cache.iter_entities():
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
        return any(_rule_matches(rule, entity_id, meta) for rule in rules)


def _rule_matches(rule: EntityRule, entity_id: str, meta: EntityMeta) -> bool:
    """Evaluate a single rule against an entity. Primitives OR together."""
    if rule.areas and meta.area_id and meta.area_id in rule.areas:
        return True
    if rule.labels and meta.labels and meta.labels.intersection(rule.labels):
        return True
    for pattern in rule.entity_globs:
        if fnmatch.fnmatchcase(entity_id, pattern):
            return True
    return bool(
        rule.device_ids and meta.device_id and meta.device_id in rule.device_ids
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool-side enforcement helpers
# ---------------------------------------------------------------------------


async def require_can_read(
    policy: AccessPolicy | None, entity_id: str, operation: str
) -> None:
    """Raise ACCESS_DENIED if ``policy`` denies reading ``entity_id``.

    No-op when ``policy`` is None (no policy active → backward compat).
    Used by tools that touch HA via code paths the client-layer gate
    doesn't cover (direct httpx, direct REST endpoints, etc.).
    """
    from .errors import create_access_denied_error, raise_tool_error

    if policy is None:
        return
    if not await policy.can_read(entity_id):
        raise_tool_error(
            create_access_denied_error(entity_id, operation=operation)
        )


async def require_can_write(
    policy: AccessPolicy | None, entity_id: str, operation: str
) -> None:
    """Raise ACCESS_DENIED if ``policy`` denies writing ``entity_id``."""
    from .errors import create_access_denied_error, raise_tool_error

    if policy is None:
        return
    if not await policy.can_write(entity_id):
        raise_tool_error(
            create_access_denied_error(entity_id, operation=operation)
        )


def require_policy_disabled(
    policy: AccessPolicy | None, tool_name: str, reason: str
) -> None:
    """Raise ACCESS_DENIED if a policy is active, refusing to run ``tool_name``.

    Used for tools whose functionality cannot be safely gated per-entity
    (e.g. Jinja template evaluation which can read any state). The caller
    is expected to document an equivalent ``tools.disabled_names`` entry
    so users can pre-empt the runtime denial.
    """
    from .errors import create_access_denied_error, raise_tool_error

    if policy is None:
        return
    raise_tool_error(
        create_access_denied_error(tool_name, operation=f"tool:{tool_name}", reason=reason)
    )


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
