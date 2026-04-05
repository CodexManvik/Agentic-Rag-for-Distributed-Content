class ToolSecurityError(PermissionError):
    pass


class ToolWhitelist:
    def __init__(self, initially_enabled: list[str] | None = None):
        self._enabled = set(initially_enabled or [])

    def enable(self, tool_name: str) -> None:
        self._enabled.add(tool_name)

    def disable(self, tool_name: str) -> None:
        self._enabled.discard(tool_name)

    def is_tool_whitelisted(self, tool_name: str) -> bool:
        return tool_name in self._enabled

    def ensure_allowed(self, tool_name: str) -> None:
        if not self.is_tool_whitelisted(tool_name):
            raise ToolSecurityError(f"Tool is not whitelisted: {tool_name}")

    def list_enabled(self) -> list[str]:
        return sorted(self._enabled)
