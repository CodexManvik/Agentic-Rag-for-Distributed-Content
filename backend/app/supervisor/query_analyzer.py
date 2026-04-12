from __future__ import annotations

import re

from app.supervisor.schemas import QueryAnalysis, QueryComplexity, QueryIntent


class QueryAnalyzer:
    _SIMPLE_PREFIXES = (
        "what is",
        "who is",
        "when did",
        "where is",
        "define",
    )
    _COMPLEX_MARKERS = ("compare", "versus", "tradeoff", "pros and cons", "then", "after")

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = (query or "").strip().lower()
        word_count = len(normalized.split())

        intent = self._infer_intent(normalized)
        complexity = self._infer_complexity(normalized, word_count)
        required_capabilities = self._infer_required_capabilities(intent, complexity, normalized)
        suggested_agents = self._infer_suggested_agents(required_capabilities)
        requires_planning = complexity in {QueryComplexity.COMPLEX, QueryComplexity.MULTI_STEP}

        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            required_capabilities=required_capabilities,
            suggested_agents=suggested_agents,
            requires_planning=requires_planning,
            confidence=0.85 if normalized else 0.5,
        )

    def _infer_intent(self, normalized: str) -> QueryIntent:
        if not normalized:
            return QueryIntent.UNKNOWN
        if normalized.startswith(("what is", "who is", "when did", "where is", "define")):
            return QueryIntent.FACT_LOOKUP
        if any(token in normalized for token in ("compare", "versus", "vs")):
            return QueryIntent.COMPARISON
        if normalized.startswith(("how to", "steps", "walk me through")):
            return QueryIntent.PROCEDURAL
        if any(token in normalized for token in ("analyze", "evaluate", "assess")):
            return QueryIntent.ANALYSIS
        if normalized.startswith(("explain", "describe", "summarize")):
            return QueryIntent.EXPLANATION
        return QueryIntent.UNKNOWN

    def _infer_complexity(self, normalized: str, word_count: int) -> QueryComplexity:
        if not normalized:
            return QueryComplexity.MODERATE
        if any(marker in normalized for marker in self._COMPLEX_MARKERS):
            return QueryComplexity.COMPLEX
        if re.search(r"\b\d+[.)]\s", normalized):
            return QueryComplexity.MULTI_STEP
        if word_count <= 10 and normalized.startswith(self._SIMPLE_PREFIXES):
            return QueryComplexity.SIMPLE_LOOKUP
        if normalized.startswith(("explain", "describe", "summarize")):
            return QueryComplexity.SIMPLE
        if word_count <= 18:
            return QueryComplexity.SIMPLE
        if word_count <= 35:
            return QueryComplexity.MODERATE
        return QueryComplexity.COMPLEX

    def _infer_required_capabilities(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        normalized: str,
    ) -> list[str]:
        caps: list[str] = ["retrieval", "synthesis"]
        if intent in {QueryIntent.ANALYSIS, QueryIntent.COMPARISON}:
            caps.append("analysis")
        if intent == QueryIntent.PROCEDURAL or complexity in {
            QueryComplexity.COMPLEX,
            QueryComplexity.MULTI_STEP,
        }:
            caps.append("planning")
        if "code" in normalized or "python" in normalized or "function" in normalized:
            caps.append("code")
        return sorted(set(caps))

    def _infer_suggested_agents(self, capabilities: list[str]) -> list[str]:
        capability_to_agent = {
            "retrieval": "retrieval",
            "synthesis": "synthesis",
            "planning": "planning",
            "analysis": "research_team",
            "code": "code_agent",
        }
        agents = [capability_to_agent[c] for c in capabilities if c in capability_to_agent]
        return sorted(set(agents))
