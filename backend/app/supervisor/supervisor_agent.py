from __future__ import annotations

from typing import Any

from app.agents.registry import AgentRegistry
from app.graph.state import AgentState
from app.session.session_state import SessionState
from app.supervisor.agent_router import AgentRouter
from app.supervisor.config import SupervisorConfig
from app.supervisor.execution_engine import ExecutionEngine
from app.supervisor.planner import TaskPlanner
from app.supervisor.query_analyzer import QueryAnalyzer
from app.supervisor.schemas import ExecutionResult, SupervisorResponse


class SupervisorAgent:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        execution_engine: ExecutionEngine,
        config: SupervisorConfig | None = None,
        analyzer: QueryAnalyzer | None = None,
        router: AgentRouter | None = None,
        planner: TaskPlanner | None = None,
    ):
        self.config = config or SupervisorConfig()
        self.registry = registry
        self.execution_engine = execution_engine
        self.analyzer = analyzer or QueryAnalyzer()
        self.router = router or AgentRouter(self.config)
        self.planner = planner or TaskPlanner(self.config)

    def process(self, query: str, session: SessionState | None = None) -> SupervisorResponse:
        analysis = self.analyzer.analyze(query)
        available_agents = self.registry.list_agents()
        routing = self.router.route(analysis, available_agents)

        seed_state = self._build_seed_state(query, session)

        plan = None
        if self.config.planning_enabled and analysis.requires_planning:
            plan = self.planner.create_plan(query, routing.selected_agents)
        else:
            plan = self.planner.create_plan(query, routing.selected_agents)

        execution = self.execution_engine.execute_plan(plan, seed_state)
        return SupervisorResponse(
            analysis=analysis,
            routing=routing,
            plan=plan,
            execution=execution,
            final_state=execution.state,
        )

    def route_state(self, state: AgentState) -> dict[str, Any]:
        query = str(state.get("query") or state.get("original_query") or "").strip()
        analysis = self.analyzer.analyze(query)
        available_agents = self.registry.list_agents()
        routing = self.router.route(analysis, available_agents)

        if state.get("final_response"):
            next_step = "FINISH"
            detail = "Routed to FINISH (response already generated)"
        elif state.get("retrieved_chunks"):
            next_step = "SynthesisAgent"
            detail = "Routed to SynthesisAgent (chunks available)"
        elif int(state.get("retries_used", 0) or 0) >= 1:
            next_step = "FINISH"
            detail = "Routed to FINISH (no chunks after retrieval attempt)"
        elif routing.selected_agents:
            first = routing.selected_agents[0]
            if first == "retrieval":
                next_step = "RetrievalAgent"
            elif first == "synthesis":
                next_step = "SynthesisAgent"
            else:
                next_step = "RetrievalAgent"
            detail = f"Routed to {next_step} (SupervisorAgent decision)"
        else:
            next_step = "FINISH"
            detail = "Routed to FINISH (no available route)"

        short_circuited = bool(routing.is_short_circuited)
        trace = list(state.get("trace", []))
        if short_circuited:
            trace.append(
                {
                    "node": "Supervisor",
                    "status": "short_circuit",
                    "detail": "Simple lookup detected, bypassing planner",
                    "complexity": analysis.complexity.value,
                }
            )
        trace.append(
            {
                "node": "Supervisor",
                "status": "ok",
                "detail": detail,
                "analysis_intent": analysis.intent.value,
                "analysis_complexity": analysis.complexity.value,
                "is_short_circuited": short_circuited,
            }
        )
        return {
            "next_step": next_step,
            "trace": trace,
            "short_circuited": short_circuited,
        }

    @staticmethod
    def _build_seed_state(query: str, session: SessionState | None) -> AgentState:
        session_agents = session.active_agents if session else []
        return {
            "query": query,
            "original_query": query,
            "selected_model": "",
            "sub_queries": [query],
            "retrieved_chunks": [],
            "final_response": "",
            "citations": [],
            "retrieval_quality": {
                "max_score": 0.0,
                "avg_score": 0.0,
                "source_diversity": 0,
                "chunk_count": 0,
                "adequate": False,
                "reason": "Not evaluated",
            },
            "retries_used": 0,
            "validation_retries_used": 0,
            "validation_errors": [],
            "used_deterministic_fallback": False,
            "abstained": False,
            "abstain_reason": None,
            "confidence": 0.0,
            "cited_indices": [],
            "synthesis_output": {
                "answer": "",
                "cited_indices": [],
                "confidence": 0.0,
                "abstain_reason": None,
            },
            "trace": [],
            "stage_timings": {},
            "stage_timestamps": {},
            "active_agents": session_agents,
        }
