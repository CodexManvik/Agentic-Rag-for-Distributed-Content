import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolManifest:
    name: str
    version: str
    description: str
    entrypoint: str
    raw: dict[str, Any]


class ToolRegistry:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._tools: dict[str, ToolManifest] = {}

    def load_tools(self) -> None:
        self._tools.clear()
        if not self.base_dir.exists():
            return

        for path in sorted(self.base_dir.rglob("*.tool.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            name = str(payload["name"])
            self._tools[name] = ToolManifest(
                name=name,
                version=str(payload.get("version", "1.0.0")),
                description=str(payload.get("description", "")),
                entrypoint=str(payload["entrypoint"]),
                raw=payload,
            )

    def list_tools(self) -> list[ToolManifest]:
        return [self._tools[name] for name in sorted(self._tools.keys())]

    def get_tool(self, name: str) -> ToolManifest:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool
