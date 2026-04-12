from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.supervisor.config import SupervisorConfig
from app.supervisor.schemas import ExecutionPlan, PlanStep


class TaskPlanner:
    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()

    def create_plan(self, query: str, agents: list[str]) -> ExecutionPlan:
        del query  # reserved for future LLM planning

        selected = agents[: self.config.max_plan_steps]
        steps: list[PlanStep] = []
        previous_step_id: str | None = None

        for index, agent_name in enumerate(selected, start=1):
            step_id = f"step-{index}"
            dependencies = [previous_step_id] if previous_step_id else []
            steps.append(
                PlanStep(
                    step_id=step_id,
                    agent=agent_name,
                    input_transform="identity",
                    output_transform="identity",
                    dependencies=dependencies,
                )
            )
            previous_step_id = step_id

        return ExecutionPlan(
            plan_id=f"plan-{uuid4()}",
            steps=steps,
            created_at=datetime.utcnow(),
            metadata={"planner": "rule_based", "planning_enabled": self.config.planning_enabled},
        )
