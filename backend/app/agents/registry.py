import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from app.agents.validator import AgentValidationError, validate_manifest_structure


@dataclass(frozen=True)
class AgentManifest:
    name: str
    version: str
    description: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    execution: dict[str, Any]
    raw: dict[str, Any]


class AgentRegistry:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._agents: dict[tuple[str, str], AgentManifest] = {}
        self._latest_by_name: dict[str, AgentManifest] = {}
        self._invalid_agents: dict[str, str] = {}

    def load_agents(self) -> None:
        self._agents.clear()
        self._latest_by_name.clear()
        self._invalid_agents.clear()

        if not self.base_dir.exists():
            logger.warning(f"Agent directory does not exist: {self.base_dir}")
            return

        for file_path in sorted(self.base_dir.rglob("*.agent.json")):
            self._load_agent_file(file_path)

    def _load_agent_file(self, file_path: Path) -> None:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            validate_manifest_structure(payload)
            manifest = AgentManifest(
                name=str(payload["name"]),
                version=str(payload["version"]),
                description=str(payload["description"]),
                inputs=dict(payload.get("inputs", {})),
                outputs=dict(payload.get("outputs", {})),
                execution=dict(payload.get("execution", {})),
                raw=payload,
            )

            key = (manifest.name, manifest.version)
            self._agents[key] = manifest

            existing = self._latest_by_name.get(manifest.name)
            if existing is None or self._version_tuple(manifest.version) >= self._version_tuple(existing.version):
                self._latest_by_name[manifest.name] = manifest
        except Exception as exc:  # keep loading other manifests
            self._invalid_agents[str(file_path)] = str(exc)
            logger.warning(f"Rejected invalid agent manifest {file_path}: {exc}")

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, int, int]:
        parts = version.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def list_agents(self) -> list[AgentManifest]:
        return [self._latest_by_name[name] for name in sorted(self._latest_by_name.keys())]

    def get_agent(self, name: str, version: str | None = None) -> AgentManifest:
        if version:
            key = (name, version)
            if key not in self._agents:
                raise AgentValidationError(f"Agent not found: {name}@{version}")
            return self._agents[key]

        agent = self._latest_by_name.get(name)
        if agent is None:
            raise AgentValidationError(f"Agent not found: {name}")
        return agent

    @property
    def invalid_agents(self) -> dict[str, str]:
        return dict(self._invalid_agents)
