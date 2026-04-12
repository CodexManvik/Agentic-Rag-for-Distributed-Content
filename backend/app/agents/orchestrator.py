from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from app.agents.context import ExecutionContext
from app.agents.registry import AgentManifest

if TYPE_CHECKING:
    from app.agents.executor import AgentExecutor


class AgentOrchestrator:
    def __init__(self, executor: "AgentExecutor"):
        self.executor = executor

    def execute_workflow(
        self,
        workflow_manifest: AgentManifest,
        state: Any,
        context: ExecutionContext,
    ) -> Any:
        execution = workflow_manifest.execution
        mode = execution.get("execution_mode", "sequence")
        steps = execution.get("workflow", [])

        if mode == "parallel":
            return self._execute_parallel(steps, state, context)
        return self._execute_sequence(steps, state, context)

    def _execute_sequence(
        self,
        steps: list[dict[str, Any]],
        state: Any,
        context: ExecutionContext,
    ) -> Any:
        current: Any = dict(state)
        for index, step in enumerate(steps):
            agent_name = str(step["agent"])
            child_context = context.child(agent_name)
            current = self.executor.execute_with_context(agent_name, current, child_context)
            self.executor.append_trace(
                current,
                node=f"workflow:{agent_name}",
                status="ok",
                detail=f"sequence step {index + 1} completed",
                context=child_context,
            )
        return current

    def _execute_parallel(
        self,
        steps: list[dict[str, Any]],
        state: Any,
        context: ExecutionContext,
    ) -> Any:
        if not steps:
            return dict(state)

        snapshots = [deepcopy(state) for _ in steps]
        contexts = [context.child(str(step["agent"])) for step in steps]

        with ThreadPoolExecutor(max_workers=len(steps)) as pool:
            futures = []
            for idx, step in enumerate(steps):
                agent_name = str(step["agent"])
                futures.append(
                    pool.submit(self.executor.execute_with_context, agent_name, snapshots[idx], contexts[idx])
                )
            results = [f.result() for f in futures]

        merged = dict(state)
        merged_trace = list(merged.get("trace", []))
        for idx, result in enumerate(results):
            for key, value in result.items():
                if key == "trace":
                    continue
                if key in {"sub_queries", "retrieved_chunks", "citations", "validation_errors", "cited_indices"}:
                    existing = list(merged.get(key, []))
                    incoming = list(value) if isinstance(value, list) else []
                    merged[key] = existing + [item for item in incoming if item not in existing]
                else:
                    merged[key] = value
            merged_trace.extend(result.get("trace", []))
            self.executor.append_trace(
                merged,
                node=f"workflow:{steps[idx]['agent']}",
                status="ok",
                detail=f"parallel branch {idx + 1} completed",
                context=contexts[idx],
            )

        merged["trace"] = merged_trace + merged.get("trace", [])[len(merged_trace):]
        return merged
