# Workflow Architecture Notes

This project runs a graph-based agentic workflow with these stages:

1. normalize_query: cleans and normalizes the incoming user query.
2. planning: produces one or more retrieval sub-queries.
3. retrieval: fetches ranked chunks from Chroma and BM25.
4. adequacy: evaluates score, diversity, and topical relevance.
5. reformulation (optional): rewrites sub-queries when evidence is weak.
6. synthesis: writes a grounded response with structured citations.
7. citation_validation: validates citation references and coverage.
8. finalize or abstain: returns final response or abstention reason.

Routing behavior:

- Weak topical overlap should route to reformulation when retries remain.
- Policy-violating queries abstain early.
- Low-latency profile uses tighter generation limits and shorter timeouts.
- Adequate evidence with citations should not end in generic abstain wording.

Retrieval priorities for workflow questions:

- Prefer project docs (README and architecture notes).
- Prefer LangGraph/LangChain documentation pages over broad marketing pages.
- Downweight generic PDF chunks unless the query is explicitly research-focused.
