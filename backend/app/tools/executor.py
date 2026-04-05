from importlib import import_module
from typing import Any, Callable, cast

from app.tools.registry import ToolRegistry
from app.tools.whitelist import ToolWhitelist


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, whitelist: ToolWhitelist):
        self.registry = registry
        self.whitelist = whitelist

    def _load_callable(self, entrypoint: str) -> Callable[..., dict[str, Any]]:
        module_name, fn_name = entrypoint.split(":", 1)
        module = import_module(module_name)
        fn = getattr(module, fn_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Invalid tool entrypoint: {entrypoint}")
        return cast(Callable[..., dict[str, Any]], fn)

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.whitelist.ensure_allowed(tool_name)
        tool = self.registry.get_tool(tool_name)
        fn = self._load_callable(tool.entrypoint)
        result = fn(args)
        if not isinstance(result, dict):
            raise ValueError("Tool result must be a dict")
        return result
