from dataclasses import dataclass


@dataclass
class SupervisorConfig:
    planning_enabled: bool = True
    max_plan_steps: int = 6
    routing_model: str = "rule_based"
    analysis_model: str = "rule_based"
    fallback_agent: str = "retrieval"
    enable_short_circuit: bool = True
    short_circuit_confidence_threshold: float = 0.7
    short_circuit_log_enabled: bool = True
