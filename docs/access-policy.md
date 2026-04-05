# Access Policy

Give each AI agent its own slice of your smart home. A policy file is a YAML
document that tells ha-mcp which entities an agent can see, which it can
change, and which tools it can call at all.

## When to use it

- **One agent per family member.** Alice's agent only touches Alice's stuff.
- **Read-only dashboards.** A briefing agent that describes the house but
  never acts on it.
- **Guest agents.** Temporary, narrow scope for visitors or contractors.
- **Safety rails.** Keep the alarm panel, locks, and whole-house climate
  out of reach of everyday agents.

## Deployment model: one MCP process per agent

Access policies apply **per process**. Each agent you want isolated gets its
own ha-mcp server process with its own `HAMCP_POLICY_FILE` env var. The MCP
client (Claude Desktop, etc.) is then pointed at that specific process.

```
                           +-------------------+
                           |  Home Assistant   |
                           +---------^---------+
                                     |  (shared HA token)
              +----------------------+----------------------+
              |                      |                      |
    +---------+---------+  +---------+---------+  +---------+---------+
    |  ha-mcp (Alice)   |  |  ha-mcp (Bob)     |  |  ha-mcp (Parents) |
    |  alice.yaml       |  |  bob.yaml         |  |  parents.yaml     |
    +---------^---------+  +---------^---------+  +---------^---------+
              |                      |                      |
        Alice's agent           Bob's agent           Parents' agent
```

There is no runtime tenant switching — isolation is by process boundary.

## Wiring it up

**1. Write a policy file.** Start with [`policies/examples/minimal.yaml`](../policies/examples/minimal.yaml)
or copy one of the worked examples.

**2. Point ha-mcp at it via `HAMCP_POLICY_FILE`.**

```bash
# stdio (Claude Desktop)
HAMCP_POLICY_FILE=./policies/alice.yaml ha-mcp

# HTTP mode
HAMCP_POLICY_FILE=./policies/alice.yaml ha-mcp-web
```

**3. Configure your MCP client to run that process.** Example Claude Desktop
entry (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "home-assistant-alice": {
      "command": "ha-mcp",
      "env": {
        "HOMEASSISTANT_URL": "http://homeassistant.local:8123",
        "HOMEASSISTANT_TOKEN": "eyJ...",
        "HAMCP_POLICY_FILE": "/Users/you/policies/alice.yaml"
      }
    }
  }
}
```

When `HAMCP_POLICY_FILE` is **unset**, no policy is applied and ha-mcp behaves
as it always has (full access). When it's **set**, the file must exist and
validate — missing/malformed/invalid files are a fatal startup error
(fail-closed).

## Fastest start: the 3-line allowlist

```yaml
version: 1
entities:
  allow:
    - entity_globs: ["light.alice_*", "media_player.alice_sonos"]
```

That's it. Everything not listed is denied. Copy, edit the patterns, done.

## Schema reference

| Field | Type | Default | Description |
|---|---|---|---|
| `version` | int | required (`1`) | Schema version. Must be exactly `1`. |
| `default_action` | `"allow"` \| `"deny"` | `"deny"` | Applied when no rule matches an entity. |
| `readonly_mode` | bool | `false` | Kill-switch: blocks every tool not marked read-only. |
| `tools.disabled_names` | list[str] | `[]` | Exact tool names to block (e.g. `ha_call_service`). |
| `tools.disabled_tags` | list[str] | `[]` | Category tags to block (e.g. `Add-ons`, `System`). |
| `entities.allow` | list[rule] | `[]` | Entities this agent can see. |
| `entities.deny` | list[rule] | `[]` | Entities this agent can never see (overrides `allow`). |
| `entities.write_allow` | list[rule] \| null | `null` | If `null`, inherits `allow`. If `[]`, blocks all writes. If non-empty, restricts writes further. |
| `entities.write_deny` | list[rule] | `[]` | Entities the agent cannot write to (even if readable). |
| `services.allow_targetless` | list[str] | `[]` | Targetless service calls (no `entity_id`/`target`) that are permitted. See [Targetless services](#targetless-services). |

### Entity rule fields

Every entry in `allow` / `deny` / `write_allow` / `write_deny` is a **rule**
with these four fields (all optional, all lists):

| Field | Matches when... |
|---|---|
| `areas` | entity's resolved area_id is in this list |
| `labels` | entity has any label in this list |
| `entity_globs` | entity_id matches any of these glob patterns (e.g. `light.alice_*`) |
| `device_ids` | entity's device_id is in this list |

Area and labels use HA's own entity→device fallback: if an entity has no area
of its own, its device's area counts. Labels are the union of the entity's
labels and its device's labels. This mirrors HA's built-in resolution.

**Unknown fields are rejected** — a typo like `entity_glob` (singular) fails
validation with the file path in the error message.

## Matching precedence

Three rules, in order:

```
 entity_id
    |
    v
+--------+   matches deny?    ---yes--> [DENY]
|  deny  | <----
+--------+
    | no
    v
+--------+   matches allow?   ---yes--> [ALLOW]
|  allow | <----
+--------+
    | no
    v
default_action                 ---deny-> [DENY]
                               ---allow-> [ALLOW]
```

**In words**: deny wins over allow; allow wins over the default.

1. If any rule in `deny` matches → **deny**.
2. Otherwise, if any rule in `allow` matches → **allow**.
3. Otherwise, apply `default_action` (defaults to `deny`).

The same three-step evaluation runs for writes, but with `write_deny` and
`write_allow` layered on top — see [Read vs write](#read-vs-write).

## OR-within-rule: the trap

**A rule matches if ANY of its primitives matches.** There is no AND across
`areas`, `labels`, `entity_globs`, or `device_ids` inside one rule.

This rule:

```yaml
allow:
  - areas: ["kitchen"]
    labels: ["owner:alice"]
```

reads as **"in kitchen OR labeled owner:alice"** — not "in kitchen AND
labeled owner:alice". It matches *every* kitchen entity AND *every*
owner:alice entity across the whole house.

To genuinely require both conditions, pick one of these:

- **Use a compound label.** Apply a single label like `alice-kitchen` in HA
  and match on that.
- **Use globs with implicit constraints.** If your entities are named
  `light.alice_kitchen_*`, a glob enforces both concepts.
- **Use `allow` + `write_allow` as layered filters.** `allow` sets the
  broad read scope; `write_allow` narrows writes to the intersection.
  See the worked example below.

**Rules ACROSS a list are also OR'd.** Two rules in `allow` both contribute —
matching either is enough.

## Read vs write

Reads use `allow`/`deny`/`default_action`. Writes use all of that **plus** two
extra checks:

```
can_write(e) = can_read(e)
             AND NOT write_deny(e)
             AND (write_allow is null OR write_allow(e))
```

The null case matters: if you omit `write_allow`, writes inherit from `allow`
(anything readable is writable). Set it to `[]` to make the view pure
read-only. Set it to a narrower rule list to restrict writes below the read
scope.

### Worked example: Alice in her bedroom and the kitchen

Alice can **see** everything in her bedroom and the kitchen, but can only
**change** things tagged as hers:

```yaml
version: 1
entities:
  allow:
    - areas: ["alice_bedroom", "kitchen"]    # read scope
  write_allow:
    - labels: ["owner:alice"]                # write scope (narrower)
```

Given these entities:

| entity_id | area | labels | can_read | can_write |
|---|---|---|---|---|
| `light.kitchen_ceiling` | kitchen | — | yes | no |
| `light.alice_desk_lamp` | alice_bedroom | `owner:alice` | yes | yes |
| `light.alice_closet` | alice_bedroom | — | yes | no |
| `light.garage` | garage | — | no | no |

Alice sees the kitchen ceiling light and her closet light but can't touch
them. She can change her own desk lamp. The garage light is invisible.

## Targetless services

Some HA service calls have no `entity_id` or `target` and cannot be gated
per-entity: `homeassistant.restart`, `notify.telegram`, `persistent_notification.create`,
`shell_command.reboot_pi`, `automation.reload`, `recorder.purge`, and so on.

Under a policy, **targetless service calls are denied by default**. Opt in
per service via `services.allow_targetless`:

```yaml
services:
  allow_targetless:
    - persistent_notification.create    # exact: one service
    - notify.*                          # wildcard: every service in a domain
```

Entries must be `domain.service` or `domain.*` (lowercase). Anything not
listed is denied — the denial response mentions `services.allow_targetless`
so the agent knows to ask for an update.

`readonly_mode: true` denies every targetless call regardless of the list —
the allow-list is only consulted when writes are permitted at all.

## Tool disabling

Some tools (backups, add-ons, YAML editing) don't have an entity_id and can't
be gated per-entity. Disable them wholesale via `tools.disabled_names` or
`tools.disabled_tags`.

Tags match the category each tool is registered under. You can disable a
whole category at once.

### Recommended "kid preset"

For a kid's agent, disable every tool category that could reconfigure the
house **or that reaches HA via a path that bypasses entity gating** (see
[Security model & limitations](#security-model--limitations) below):

```yaml
tools:
  disabled_tags:
    # Config/reconfiguration surface
    - "Add-ons"
    - "HACS"
    - "Files"
    - "System"
    - "Integrations"
    - "Entity Registry"
    - "Device Registry"
    - "Labels & Categories"
    # These are auto-gated per-entity now, but disabling keeps them out
    # of the agent's tool list entirely (cleaner UX):
    - "History & Statistics"
    - "Camera"
    - "Calendar"
  disabled_names:
    - ha_config_set_yaml       # raw YAML edits bypass every entity rule
    - ha_eval_template         # Jinja can read any entity (auto-denied too)
    - ha_get_integration       # OAuth tokens in config entries (auto-denied too)

services:
  # Targetless calls are denied by default — opt in the ones you need:
  allow_targetless:
    - persistent_notification.create
```

This leaves day-to-day control (lights, media players, climate, todo lists,
scenes, scripts, automations within the kid's entity scope) while closing
off the tools that could undo the policy or leak state from entities outside
the allow list.

## Security model & limitations

Policies are enforced at **two layers**, and you need both for a tight
isolation:

1. **Client layer (automatic):** The policy gates every call through
   `get_states`, `get_entity_state`, `set_entity_state`, `call_service`,
   and the logbook REST endpoint, plus WebSocket responses from
   `entity_registry/*`, `device_registry/list`, `homeassistant/expose_entity/list`,
   and `zone/list`. Entity-scoped WS commands (`todo/item/*`,
   `homeassistant/expose_entity`) are pre-gated before dispatch. Targetless
   service calls (no `entity_id`/`target`) are denied unless explicitly
   allow-listed in `services.allow_targetless` — see
   [Targetless services](#targetless-services). This covers the bulk of
   MCP tools — search, state reads, service calls, automations, scripts,
   helpers, todo lists, voice-assistant exposure, etc.

2. **Tool layer (automatic for known bypasses):** Some tools reach Home
   Assistant through code paths the client layer doesn't cover (direct
   httpx, direct `_request`, direct `ws_client`). Those tools now carry
   their own per-entity gate or are hard-disabled under an active policy:

   | Path | Tools | Status |
   |---|---|---|
   | WS `history/*` & `recorder/statistics_*` | `ha_get_history`, `ha_get_statistics` | Gated per-entity at tool layer — deny whole call if ANY requested entity is outside scope |
   | Jinja templates | `ha_eval_template` | Hard-disabled under any active policy (Jinja reads any state via `states('lock.front')`) — add to `tools.disabled_names` to pre-empt the runtime denial |
   | Direct httpx (camera images) | `ha_get_camera_image` | Gated per-entity at tool layer |
   | Unfiltered REST (calendar) | `ha_config_get_calendar_events` | Gated per-entity at tool layer (write tools already flow through the service-call gate) |
   | Integration config dump | `ha_get_integration` | Hard-disabled under any active policy (config entries expose OAuth tokens and provider credentials) — add to `tools.disabled_names` to pre-empt the runtime denial |

   For the hard-disabled tools, the kid preset still recommends listing
   them under `tools.disabled_names` so the tool never shows up in the
   agent's tool list in the first place.

**Principle: least privilege via both layers.** If a tool category isn't
explicitly gated at the entity level and you aren't sure what it can reach,
disable it. The kid preset above already includes these categories.

### `readonly_mode` as a shortcut

Setting `readonly_mode: true` blocks **every tool that isn't explicitly
marked `readOnlyHint: true`**. This is stricter than `write_allow: []`
because it also gates entity-less tools like `ha_call_service` and
`ha_backup_create`. Use it for monitoring/viewer agents.

## Troubleshooting

### "Entity is outside your policy scope" / `ACCESS_DENIED`

The agent tried to read or write an entity that the policy denies. The
error response includes the entity_id in its `context` and a suggestion
to check `HAMCP_POLICY_FILE`. Fix by:

1. Adding the entity (or its area/label/glob) to `entities.allow`.
2. Removing a too-broad `entities.deny` rule.
3. Switching `default_action` to `allow` if you want a permissive baseline.

### Policy file errors at startup

If `HAMCP_POLICY_FILE` points to a file that doesn't exist, isn't valid
YAML, or fails schema validation, ha-mcp refuses to start. The error names
the file and the specific problem. Check:

- **File path** — is it absolute, or relative to where the process starts?
- **Field names** — typos like `entity_glob` (singular) are rejected.
- **Version** — must be `version: 1`.
- **`default_action`** — only `"allow"` or `"deny"`.

### Hidden entity I expected to see

1. Is it denied by an `entities.deny` rule? (Deny wins.)
2. Does it fall through to `default_action: deny`? Add it to `allow`.
3. Is its area/label resolved via the device? Labels and area_id are
   unioned across entity + device — check both in HA's UI.

### Entity is readable but writes fail

1. Is `readonly_mode: true`? All writes blocked.
2. Is `write_deny` matching? Overrides `write_allow`.
3. Is `write_allow` set to a narrower list that doesn't include this
   entity? Either widen `write_allow` or set it to `null` (omit the key)
   to inherit `allow`.

## Example policies

Ready-to-copy starter templates live in [`policies/examples/`](../policies/examples/):

| File | Purpose |
|---|---|
| [`minimal.yaml`](../policies/examples/minimal.yaml) | 3-line pure allowlist — the fastest start. |
| [`alice.yaml`](../policies/examples/alice.yaml) | Kid agent: scoped read + narrower write, full kid-preset tool disables. |
| [`bob.yaml`](../policies/examples/bob.yaml) | Mirror of alice.yaml — shows the copy+rename pattern. |
| [`parents.yaml`](../policies/examples/parents.yaml) | Permissive (`default_action: allow`) with carve-outs for kids' devices and backup restore. |
| [`readonly_viewer.yaml`](../policies/examples/readonly_viewer.yaml) | `readonly_mode: true` monitoring agent — full read, zero writes. |
