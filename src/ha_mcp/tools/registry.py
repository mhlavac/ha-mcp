"""
Tools registry for Smart MCP Server - manages registration of all MCP tools.

This module uses lazy auto-discovery to find and register all tool modules.
Tool modules are discovered at startup but only imported when first accessed,
improving server startup time significantly (especially for binary distributions).

Adding a new tools module is simple:
1. Create tools_*.py file with a register_*_tools(mcp, client, **kwargs) function
2. The function will be auto-discovered and registered lazily

No changes to this file are needed when adding new tool modules!

Tool filtering:
Set ENABLED_TOOL_MODULES environment variable to filter which tools are loaded:
- "all" (default): Load all tools
- "automation": Load only automation-related tools (automations, scripts, traces, blueprints)
- Comma-separated list: Load specific modules (e.g., "tools_config_automations,tools_search")
"""

import logging
import pkgutil
from pathlib import Path
from typing import Any

from ..access_policy import AccessPolicy, get_global_policy

logger = logging.getLogger(__name__)


class PolicyAwareMCP:
    """Thin proxy around a FastMCP server that gates ``.tool()`` registrations.

    When a tool decorator is applied, the wrapped function is passed through
    unchanged if the active policy rejects it (name/tag/readonly check).
    All other attributes (``.resource``, ``.prompt``, ``.add_transform``, etc.)
    are delegated to the underlying MCP server unchanged.
    """

    def __init__(self, mcp: Any, policy: AccessPolicy):
        self._mcp = mcp
        self._policy = policy

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept the ``.tool()`` decorator to skip policy-denied tools.

        Mirrors FastMCP's decorator signature: ``@mcp.tool(name=..., tags=...,
        annotations=...)`` — the tool name defaults to the function name.
        """
        tags = kwargs.get("tags")
        annotations = kwargs.get("annotations")
        explicit_name = kwargs.get("name")

        def decorator(fn: Any) -> Any:
            resolved_name = explicit_name or fn.__name__
            tag_set: set[str] | None = set(tags) if tags else None
            if not self._policy.is_tool_allowed(resolved_name, tag_set, annotations):
                logger.info("Policy skipped tool: %s", resolved_name)
                return fn
            return self._mcp.tool(*args, **kwargs)(fn)

        return decorator

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (resource, prompt, add_transform, run_async, ...)
        return getattr(self._mcp, name)

# Modules that don't follow the tools_*.py naming convention
# These are handled explicitly for backward compatibility
EXPLICIT_MODULES = {
    "backup": "register_backup_tools",
}

# Preset module groups for common use cases
MODULE_PRESETS = {
    "automation": [
        "tools_config_automations",
        "tools_config_scripts",
        "tools_traces",
        "tools_blueprints",
        "tools_search",  # Useful for finding entities in automations
    ],
}


class ToolsRegistry:
    """Manages registration of all MCP tools for the smart server.

    Implements lazy loading pattern: tool modules are discovered at startup
    but only imported and registered when the server starts accepting connections.
    This significantly improves startup time for binary distributions.

    Tool filtering is controlled via ENABLED_TOOL_MODULES environment variable:
    - "all": Load all tools (default)
    - "automation": Load automation-related tools only
    - Comma-separated list: Load specific modules
    """

    def __init__(self, server: Any, enabled_modules: str = "all") -> None:
        self.server = server
        self.client = server.client
        self.mcp = server.mcp
        self._enabled_modules = enabled_modules
        # These are now lazily initialized via server properties
        self._smart_tools = None
        self._device_tools = None
        self._modules_registered = False
        # Discover modules at init time (fast - no imports)
        self._discovered_modules = self._discover_tool_modules()

    @property
    def smart_tools(self) -> Any:
        """Lazily get smart_tools from server."""
        if self._smart_tools is None:
            self._smart_tools = self.server.smart_tools
        return self._smart_tools

    @property
    def device_tools(self) -> Any:
        """Lazily get device_tools from server."""
        if self._device_tools is None:
            self._device_tools = self.server.device_tools
        return self._device_tools

    def _get_enabled_module_list(self) -> set[str] | None:
        """Parse enabled_modules config into a set of module names.

        Returns None if all modules should be enabled.
        """
        if self._enabled_modules.lower() == "all":
            return None

        # Check for preset names
        if self._enabled_modules.lower() in MODULE_PRESETS:
            return set(MODULE_PRESETS[self._enabled_modules.lower()])

        # Parse comma-separated list
        modules = {m.strip() for m in self._enabled_modules.split(",") if m.strip()}
        return modules if modules else None

    def _discover_tool_modules(self) -> list[str]:
        """Discover tool module names without importing them.

        This is a fast operation that only reads file names.
        Returns list of module names that follow the tools_*.py convention,
        filtered by ENABLED_TOOL_MODULES configuration.
        """
        enabled_set = self._get_enabled_module_list()
        discovered = []
        package_path = Path(__file__).parent

        for module_info in pkgutil.iter_modules([str(package_path)]):
            module_name = module_info.name
            if module_name.startswith("tools_"):
                # Filter if enabled_set is specified
                if enabled_set is None or module_name in enabled_set:
                    discovered.append(module_name)

        # Add explicit modules (only if enabled or no filter)
        discovered.extend(
            module_name for module_name in EXPLICIT_MODULES
            if enabled_set is None or module_name in enabled_set
        )

        if enabled_set is not None:
            logger.info(
                f"Tool filtering active: {len(discovered)} modules enabled "
                f"(filter: {self._enabled_modules})"
            )
        else:
            logger.debug(f"Discovered {len(discovered)} tool modules (not yet imported)")

        return discovered

    def register_all_tools(self) -> None:
        """Register all tools with the MCP server using lazy auto-discovery.

        Tool modules are imported and registered only when this method is called,
        which happens after the MCP server is ready to accept connections.
        """
        if self._modules_registered:
            logger.debug("Tools already registered, skipping")
            return

        import importlib

        # Build kwargs with all available dependencies (lazy access)
        kwargs = {
            "smart_tools": self.smart_tools,
            "device_tools": self.device_tools,
        }

        # Wrap MCP with policy-aware proxy if a policy is active. When there is
        # no policy, tool registration goes straight through to FastMCP unchanged.
        policy = get_global_policy()
        mcp_for_registration: Any = (
            PolicyAwareMCP(self.mcp, policy) if policy is not None else self.mcp
        )

        registered_count = 0

        # Import and register tools_*.py modules
        for module_name in self._discovered_modules:
            # Skip explicit modules - handled separately
            if module_name in EXPLICIT_MODULES:
                continue

            try:
                module = importlib.import_module(f".{module_name}", "ha_mcp.tools")

                # Find the register function (convention: register_*_tools)
                register_func = None
                for attr_name in dir(module):
                    if attr_name.startswith("register_") and attr_name.endswith("_tools"):
                        register_func = getattr(module, attr_name)
                        break

                if register_func:
                    register_func(mcp_for_registration, self.client, **kwargs)
                    registered_count += 1
                    logger.debug(f"Registered tools from {module_name}")
                else:
                    logger.warning(
                        f"Module {module_name} has no register_*_tools function"
                    )

            except Exception as e:
                logger.error(f"Failed to register tools from {module_name}: {e}")
                raise

        # Register explicit modules (those not following tools_*.py convention)
        # Only register if they were included in discovered modules (respects filtering)
        for module_name, func_name in EXPLICIT_MODULES.items():
            if module_name not in self._discovered_modules:
                continue
            try:
                module = importlib.import_module(f".{module_name}", "ha_mcp.tools")
                register_func = getattr(module, func_name)
                register_func(mcp_for_registration, self.client, **kwargs)
                registered_count += 1
                logger.debug(f"Registered tools from {module_name}")
            except Exception as e:
                logger.error(f"Failed to register tools from {module_name}: {e}")
                raise

        self._modules_registered = True
        logger.info(f"Auto-discovery registered tools from {registered_count} modules")
