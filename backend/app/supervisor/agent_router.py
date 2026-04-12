from __future__ import annotations

from app.agents.registry import AgentManifest
from app.supervisor.config import SupervisorConfig
from app.supervisor.schemas import QueryAnalysis, QueryComplexity, RoutingDecision, RoutingStep


class AgentRouter:
    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()

    def route(self, analysis: QueryAnalysis, available_agents: list[AgentManifest]) -> RoutingDecision:
        available_names = {a.name for a in available_agents}

        if (
            self.config.enable_short_circuit
            and analysis.complexity == QueryComplexity.SIMPLE_LOOKUP
            and {"retrieval", "synthesis"}.issuperset(set(analysis.required_capabilities))
            and "retrieval" in available_names
        ):
            selected = ["retrieval", "synthesis"] if "synthesis" in available_names else ["retrieval"]
            return RoutingDecision(
                selected_agents=selected,
                execution_order=[
                    RoutingStep(agent=selected[0], order=1),
                    *([RoutingStep(agent=selected[1], order=2)] if len(selected) > 1 else []),
                ],
                reason="Short-circuit routing: simple lookup query",
                is_short_circuited=True,
            )

        selected_agents = [agent for agent in analysis.suggested_agents if agent in available_names]
        if not selected_agents:
            fallback = self.config.fallback_agent if self.config.fallback_agent in available_names else None
            if fallback:
                selected_agents = [fallback]

        execution_order: list[RoutingStep] = [
            RoutingStep(agent=agent_name, order=i + 1) for i, agent_name in enumerate(selected_agents)
        ]

        return RoutingDecision(
            selected_agents=selected_agents,
            execution_order=execution_order,
            reason="Rule-based capability routing",
            is_short_circuited=False,
        )
