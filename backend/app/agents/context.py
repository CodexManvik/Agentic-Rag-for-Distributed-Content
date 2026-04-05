from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True)
class ExecutionContext:
    execution_id: str
    parent_execution_id: str | None
    root_execution_id: str
    depth: int
    path: tuple[str, ...] = field(default_factory=tuple)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def root(agent_name: str) -> "ExecutionContext":
        execution_id = str(uuid4())
        return ExecutionContext(
            execution_id=execution_id,
            parent_execution_id=None,
            root_execution_id=execution_id,
            depth=0,
            path=(agent_name,),
        )

    def child(self, agent_name: str) -> "ExecutionContext":
        return ExecutionContext(
            execution_id=str(uuid4()),
            parent_execution_id=self.execution_id,
            root_execution_id=self.root_execution_id,
            depth=self.depth + 1,
            path=(*self.path, agent_name),
        )
