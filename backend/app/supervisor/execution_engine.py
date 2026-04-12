from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.agents.context import ExecutionContext
from app.agents.executor import AgentExecutor
from app.graph.state import AgentState
from app.supervisor.schemas import ExecutionPlan, ExecutionResult


class ExecutionEngine:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor

    def execute_plan(self, plan: ExecutionPlan, state: AgentState) -> ExecutionResult:
        current_state: dict[str, Any] = deepcopy(dict(state))
        executed_steps: list[str] = []
        errors: list[str] = []

        root = ExecutionContext.root("supervisor_execution")

        for step in plan.steps:
            try:
                child_context = root.child(step.agent)
                result = self.executor.execute_with_context(
                    step.agent,
                    current_state,
                    child_context,
                )
                current_state = dict(result)
                executed_steps.append(step.step_id)
            except Exception as exc:
                errors.append(f"{step.step_id}:{step.agent}:{exc}")
                return ExecutionResult(
                    success=False,
                    state=current_state,
                    executed_steps=executed_steps,
                    errors=errors,
                )

        return ExecutionResult(
            success=True,
            state=current_state,
            executed_steps=executed_steps,
            errors=errors,
        )
