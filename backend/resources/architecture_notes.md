# Smart Knowledge Navigator - Architecture & Workflow Design

**Last Updated:** 2026-03-29  
**Purpose:** Technical design documentation for the agentic RAG workflow

## Executive Summary

Smart Knowledge Navigator implements an 8-node **LangGraph state machine** that orchestrates multiple specialized agents to provide citation-backed answers with explainable reasoning. The architecture prioritizes **groundedness over fluency**, **transparency over speed**, and **abstention over hallucination**.

**Core Innovation:** Dynamic evidence adequacy evaluation with bounded reformulation retries and multi-stage citation validation.

---

## Workflow Architecture

### Agent Pipeline Overview

The system implements a directed acyclic graph (DAG) with conditional routing:

```
normalize_query → planning → retrieval → adequacy ──→ synthesis → citation_validation ──→ finalize
                                            │                                    │
                                            └──→ reformulation ──→ (loop)       └──→ abstain
                                            │
                                            └──→ abstain
```

### Node Descriptions

#### 1. **normalize_query** (Preprocessor)
**Purpose:** Clean and expand user queries for better retrieval  
**Operations:**
- Strip whitespace, punctuation normalization
- Abbreviation expansion (RAG → retrieval augmented generation)
- Lowercase conversion for consistency
**Output:** Normalized query string  
**Duration:** ~10-50ms

#### 2. **planning** (Query Expansion Agent)
**Purpose:** Generate multiple retrieval sub-queries to capture different aspects  
**Behavior by Profile:**
- `low_latency`: 3 sub-queries, rule-based expansion (comparison splits, procedural variants)
- `balanced`: 4 sub-queries, hybrid rules + semantic variants
- `high_quality`: 5 sub-queries, full LLM-powered planning

**Strategies:**
- **Comparison queries**: Split "A vs B" into separate lookups for A and B
- **Procedural queries**: Add "tutorial steps", "best practices" variants
- **Definitional queries**: Add "overview", "use cases examples" expansions
- **Generic queries**: Add "guide", "documentation" suffixes

**Output:** List of 3-5 sub-queries  
**Duration:** ~500-1,500ms (LLM call for balanced/high_quality)

#### 3. **retrieval** (Hybrid Search Agent)
**Purpose:** Fetch relevant document chunks using vector + lexical search  
**Implementation:**
- **Vector Search**: ChromaDB with nomic-embed-text embeddings (768-dim)
- **BM25 Search**: Lexical matching with cached tokenization
- **Hybrid Scoring**: 0.7 × vector_score + 0.3 × BM25_score
- **Reranking**: Optional semantic reranking (disabled by default for speed)

**Per-Query Retrieval:**
- Top-k per sub-query: 3-5 chunks (profile-dependent)
- Final top-k: 4-8 chunks (deduplicated by chunk_id)
- Min score threshold: 0.42 (balanced), 0.35 (high_quality)

**Metadata Enrichment:**
Each chunk includes:
- `source`: Original URL/path
- `source_type`: web | pdf_url | pdf_local | project_doc | confluence
- `title`: Document title
- `section`: Heading/section name
- `page_number`: PDF page (if applicable)
- `url`: Canonical URL
- `content_hash`: SHA-256 for deduplication
- `ingestion_timestamp`: ISO 8601 timestamp

**Special Boosting:**
- Project docs (README, architecture_notes): 1.35× score multiplier for workflow queries
- LangGraph/LangChain docs: 1.2× multiplier for agent workflow queries
- PDF penalty for workflow queries: 0.55× (downweight academic papers)

**Output:** Ranked list of RetrievedChunk objects  
**Duration:** ~800-2,000ms (depends on corpus size)

#### 4. **adequacy** (Evidence Quality Evaluator)
**Purpose:** Assess if retrieved evidence is sufficient to answer the query  
**Scoring Factors:**

1. **Score Threshold**: Max score ≥ 0.42 (balanced) or ≥ 0.35 (high_quality)
2. **Chunk Count**: ≥ 2-3 chunks (profile-dependent)
3. **Source Diversity**: ≥ 1-3 unique sources (query difficulty-dependent)
4. **Topical Overlap**: Query terms present in retrieved chunks (≥10% overlap)
5. **Entity Overlap**: Named entities/key phrases match (≥10% overlap)

**Special Cases:**
- **Policy Violations**: Early abstain if query triggers policy guard (private/confidential/adversarial patterns)
- **Zero Evidence**: Immediate abstain if no chunks retrieved
- **Hard Queries**: Relaxed thresholds (+0.08 score boost, +2 source diversity requirement)
- **Low-Latency Mode**: Skip overlap checks for speed

**Routing Decisions:**
- `adequate=true` → route to **synthesis**
- `adequate=false` + retries remaining → route to **reformulation**
- `adequate=false` + no retries → route to **abstain**

**Output:** RetrievalQuality object with adequacy boolean and reason  
**Duration:** ~50-200ms

#### 5. **reformulation** (Query Refinement Agent)
**Purpose:** Rewrite sub-queries when initial retrieval yields weak evidence  
**Trigger Conditions:**
- Weak topical overlap detected
- Low diversity (single source retrieved)
- Max retries not exhausted (default: 1 retry)

**Strategies:**
- Add domain-specific terms (e.g., "LangGraph" → "LangGraph state graph node edges")
- Broaden scope (e.g., "qwen model" → "ollama qwen model configuration")
- Narrow focus (e.g., "RAG" → "RAG retrieval pipeline implementation")

**Bounded Retries:**
- `max_retrieval_retries`: 0 (low_latency), 1 (balanced), 2 (high_quality)
- Prevents infinite loops and excessive latency

**Output:** Updated sub_queries list  
**Duration:** ~500-1,000ms (LLM call)

#### 6. **synthesis** (Answer Generation Agent)
**Purpose:** Generate structured JSON response with citations  
**Prompt Structure:**
```
CONTEXT: [top 4-8 chunks with [1], [2], ... indices]

QUERY: {user_query}

INSTRUCTIONS:
- Answer using ONLY information from CONTEXT
- Cite sources with [n] notation
- Return JSON: {"answer": "...", "cited_indices": [1,2], "confidence": 0.0-1.0}
- If uncertain, set abstain_reason
```

**Constraints:**
- Max output tokens: 512 (low_latency), 800 (balanced), 1000 (high_quality)
- Timeout: 25s (low_latency), 45s (balanced), 90s (high_quality)
- Stop sequences: `</think>`, triple backticks, prompt injection markers
- Temperature: 1.0 (balanced creativity), Top-P: 0.95

**Structured Output Parsing:**
- Expects valid JSON with `answer`, `cited_indices`, `confidence`, `abstain_reason`
- Fallback parser for malformed JSON (extract text between braces)
- Strict retry on parse failure with explicit JSON schema reminder

**Output:** SynthesisOutput TypedDict  
**Duration:** ~3,000-8,000ms (LLM generation, largest contributor to latency)

#### 7. **citation_validation** (Guardrails Agent)
**Purpose:** Verify that citations are factually grounded in source chunks  
**Validation Steps:**

1. **Index Validity**: All cited indices exist in citation array
2. **Coverage**: Factual sentences have at least one citation
3. **Semantic Grounding**: Citation content overlaps with claim (≥15% token overlap)

**Sentence Classification:**
- **Factual**: Contains verbs, entities, numbers, or >5 words
- **Connective**: "however", "also", "therefore" (exempt from citation)
- **Short phrases**: ≤5 words (exempt unless contains verbs/entities)

**Regeneration Policy:**
- **First failure**: Retry synthesis with stricter prompt ("Every factual claim MUST have [n]")
- **Second failure**: Hard abstain with reason "citation_validation_failed"

**Error Categories Tracked:**
- `empty_answer`: No text generated
- `no_citations`: Answer has no [n] markers
- `invalid_index`: Citation index out of bounds
- `low_coverage`: <40% of sentences cited
- `semantic_mismatch`: Claim not supported by cited chunk

**Output:** Updated abstained boolean, validation_errors list  
**Duration:** ~100-300ms (no LLM calls, pure validation logic)

#### 8a. **finalize** (Success Path)
**Purpose:** Package final response for API return  
**Operations:**
- Build citation objects with source metadata
- Calculate final confidence score
- Format trace events with duration
- Set abstained=false

**Output:** Complete NavigatorState  
**Duration:** ~10-50ms

#### 8b. **abstain** (Failure Path)
**Purpose:** Explain why the system cannot answer  
**Abstain Reasons:**
- `Policy scope guard triggered` (private/confidential query)
- `No relevant chunks found` (retrieval returned 0 results)
- `Evidence is insufficient or unverifiable` (adequacy check failed)
- `Citation validation failed after retry` (grounding violations)
- `Synthesis parse failure after strict retry` (malformed JSON from model)

**Output:** NavigatorState with abstained=true, abstain_reason set  
**Duration:** ~10-50ms

---

## State Management

### NavigatorState Schema (TypedDict)

```python
{
    "query": str,                           # Normalized query
    "original_query": str,                  # Raw user input
    "sub_queries": list[str],               # Generated retrieval queries
    "retrieved_chunks": list[RetrievedChunk],
    "final_response": str,
    "citations": list[Citation],
    "retrieval_quality": RetrievalQuality,
    "retries_used": int,                    # Reformulation attempts
    "validation_retries_used": int,         # Citation validation retries
    "validation_errors": list[str],
    "abstained": bool,
    "abstain_reason": str | None,
    "confidence": float,
    "cited_indices": list[int],
    "synthesis_output": SynthesisOutput,
    "trace": list[TraceEvent],              # Node execution log
    "stage_timings": dict[str, float],      # Cumulative ms per stage
    "stage_timestamps": dict[str, dict],    # Detailed timing metadata
}
```

**State Transitions:**
- State is **immutable per node** (functional updates)
- Each node returns updated state copy
- Conditional edges inspect state to determine routing

---

## Routing Logic

### After Adequacy Check

```python
def _route_after_adequacy(state: NavigatorState) -> str:
    if state["abstained"]:
        return "abstain"  # Policy violation caught
    
    quality = state["retrieval_quality"]
    
    # Check for weak evidence with retry budget
    weak_topical = "weak topical match" in quality["reason"].lower()
    if weak_topical and state["retries_used"] < max_retries:
        return "reformulation"
    
    # Route to synthesis if adequate
    if quality["adequate"]:
        return "synthesis"
    
    # Check for minimal meaningful evidence
    has_meaningful_evidence = (
        quality["max_score"] >= 0.30 and
        quality["chunk_count"] >= 2 and
        quality["source_diversity"] >= 2
    )
    
    if has_meaningful_evidence:
        return "synthesis"  # Try synthesis with minimal evidence
    
    # Low-latency retry before giving up
    if profile == "low_latency" and state["retries_used"] < max_retries:
        return "reformulation"
    
    return "abstain"
```

### After Citation Validation

```python
def _route_after_validation(state: NavigatorState) -> str:
    return "abstain" if state["abstained"] else "finalize"
```

---

## Configuration & Tuning

### Runtime Profile Comparison

| Parameter                  | low_latency | balanced | high_quality |
| -------------------------- | ----------: | -------: | -----------: |
| Planner max sub-queries    |           3 |        4 |            5 |
| Retrieval per-query k      |           3 |        4 |            4 |
| Retrieval top-k            |         5-6 |        6 |            8 |
| Context chunk limit        |           4 |        6 |            8 |
| Max retrieval retries      |           0 |        1 |            2 |
| Max validation retries     |           0 |        1 |            1 |
| Model timeout (planning)   |          6s |       6s |           6s |
| Model timeout (synthesis)  |         25s |      45s |          90s |
| Max output tokens          |         512 |      800 |        1,000 |
| Skip overlap check         |        true |    false |        false |

### Key Thresholds (Balanced Profile)

```python
RETRIEVAL_MIN_SCORE = 0.42              # Vector+BM25 hybrid score
RETRIEVAL_MIN_CHUNKS = 3                # Minimum chunks for adequacy
RETRIEVAL_MIN_SOURCE_DIVERSITY = 2      # Unique sources required
RETRIEVAL_QUERY_OVERLAP_MIN = 0.10      # Term overlap threshold
RETRIEVAL_ENTITY_OVERLAP_MIN = 0.10     # Named entity overlap
CITATION_SEMANTIC_OVERLAP_MIN = 0.15    # Claim-to-source overlap
CHUNK_SIZE = 1200                       # Characters per chunk
CHUNK_OVERLAP = 200                     # Overlap for context preservation
```

---

## Special Behaviors & Edge Cases

### Workflow Query Detection
Queries about "LangGraph", "agent workflow", "state machine" trigger:
- Project doc boost (README, architecture_notes): 1.35× score
- LangGraph doc boost: 1.2× score
- PDF penalty: 0.55× (downweight academic papers)
- Higher term overlap requirement (≥2 terms instead of ≥1)

**Rationale:** The system should prefer its own documentation when answering meta-questions about its architecture.

### Hard Query Handling
Queries flagged as "hard" (multi-hop, comparison, edge-case ambiguity) get:
- Score boost: +0.08 (relaxed threshold to 0.34 instead of 0.42)
- Higher source diversity: 3 sources required instead of 2
- More retries: +1 additional reformulation attempt

**Detection:** Bucket metadata from evaluation dataset or heuristics (multiple entities, "compare", "difference")

### Policy Guard Triggers
30+ patterns detect adversarial/private queries:
- **Private**: "confidential", "internal", "non-public", "employee"
- **Credentials**: "api key", "password", "secret", "leak"
- **Prompt Injection**: "ignore previous", "bypass your", "reveal internal", "dump all"
- **Sensitive**: "salary band", "hr policy", "payroll"

**Action:** Immediate abstain with `"Policy scope guard triggered"` reason, no retrieval performed.

### Low-Latency Optimizations
- Skip topical overlap check (saves ~50-100ms)
- Use rule-based planning instead of LLM (saves ~500ms)
- Reduce context from 6 to 4 chunks (saves ~200ms in synthesis)
- Shorter timeout with aggressive truncation (saves ~20s on slow generations)
- Zero reformulation retries (saves ~2-3s on weak evidence)

**Trade-off:** ~15-25s latency reduction, ~5% accuracy drop on hard queries

---

## Observability & Debugging

### Trace Events
Every node logs:
```json
{
  "node": "retrieval",
  "status": "ok",
  "detail": "Retrieved 6 chunks, max_score=0.87, diversity=3",
  "ts": "2026-03-29T17:30:00.000Z",
  "duration_ms": 1250.5
}
```

### Stage Timings
Cumulative duration tracking per stage:
```json
{
  "planning": 850.5,
  "retrieval": 1200.3,
  "adequacy": 120.8,
  "synthesis": 3500.2,
  "citation_validation": 180.4,
  "finalize": 25.1
}
```

### Stage Timestamps (Detailed)
Includes retry attempts:
```json
{
  "synthesis": {
    "started_at": "2026-03-29T17:30:05Z",
    "finished_at": "2026-03-29T17:30:12Z",
    "duration_ms": 7200.5,
    "attempt_count": 2,
    "last_attempt_duration_ms": 3800.2,
    "attempts": [
      {"started_at": "...", "duration_ms": 3400.3},
      {"started_at": "...", "duration_ms": 3800.2}
    ]
  }
}
```

### Latency Logging
Append-only JSONL log (`backend/resources/stage_latency.jsonl`):
```json
{
  "ts": "2026-03-29T17:30:00Z",
  "query": "How does LangGraph support adaptive workflows?",
  "profile": "balanced",
  "abstained": false,
  "adequate": true,
  "chunk_count": 6,
  "sub_queries": ["LangGraph adaptive workflows", "state graph routing"],
  "top_chunks": [{"source": "...", "score": 0.92}],
  "stage_timings": {...}
}
```

---

## Design Decisions & Rationale

### Why Abstention Over Guessing?
**Decision:** Hard abstain when evidence is weak or citations invalid  
**Rationale:** Enterprise use cases (legal, medical, financial) require high precision over recall. A "I don't know" is safer than a plausible-sounding hallucination.  
**Trade-off:** Lower answer rate (~60-70% on hard queries) but near-zero hallucination rate.

### Why Bounded Retries?
**Decision:** Max 1-2 reformulation attempts, hard stop after that  
**Rationale:** Prevents infinite loops and excessive latency. If 2 rounds of reformulation don't work, the answer likely isn't in the corpus.  
**Trade-off:** Some answerable queries get abstained after retries exhausted.

### Why Hybrid Retrieval (Vector + BM25)?
**Decision:** 70% vector, 30% BM25 weighted scoring  
**Rationale:** Vector search excels at semantic similarity but misses exact term matches (e.g., "API key", "LangGraph"). BM25 provides keyword precision.  
**Evidence:** Hybrid retrieval improved Hit@k by 12% vs vector-only in early experiments.

### Why Citation Validation Post-Generation?
**Decision:** Generate first, validate second (with retry)  
**Rationale:** Constrained generation ("cite as you write") reduces fluency and increases parse failures. Post-validation with regeneration balances quality and grounding.  
**Trade-off:** Extra synthesis call on validation failure (~3-5s latency penalty).

### Why Three Runtime Profiles?
**Decision:** low_latency, balanced, high_quality  
**Rationale:** Different use cases need different trade-offs:
- **low_latency**: Live demos, chatbot UX (30-40s P50)
- **balanced**: Production default, good quality/speed (40-50s P50)
- **high_quality**: Research, legal review, offline batch (60-90s P50)

---

## Known Limitations & Future Work

### 1. Small Model Limitations
**Current:** llama3.2:3b struggles with multi-hop reasoning and citation discipline  
**Impact:** 16.3% citation precision, 0% Hit@k on comparison queries  
**Mitigation:** Use qwen3.5:4b (requires 6GB VRAM) or llama3.2:8b (8GB VRAM)  
**Future:** Implement Chain-of-Thought prompting or reasoning traces

### 2. Single-Stage Synthesis
**Current:** All synthesis happens in one LLM call  
**Limitation:** Cannot combine facts from 5+ sources requiring intermediate reasoning  
**Future:** Implement MapReduce or hierarchical summarization for complex queries

### 3. Static Adequacy Thresholds
**Current:** Fixed thresholds (score ≥0.42, chunks ≥3, diversity ≥2)  
**Limitation:** Not adaptive to query complexity or corpus characteristics  
**Future:** Learn thresholds from eval feedback or use learned adequacy predictor

### 4. No Cross-Document Entity Resolution
**Current:** Treats each chunk independently  
**Limitation:** Cannot resolve "it", "this", "the system" across documents  
**Future:** Add coreference resolution or entity linking pass

### 5. Limited Multi-Modal Support
**Current:** Text-only (web, PDF text extraction)  
**Limitation:** Cannot process images, tables, diagrams  
**Future:** Add vision models for diagram/chart QA, table parsing

---

## Performance Characteristics

**Measured on:** Windows 10, AMD Ryzen 9 5900X, 32GB RAM, Ollama with llama3.2:3b

- **Cold start**: 10-15s (BM25 cache build + model loading)
- **Warm query P50**: 35s (balanced), 25s (low_latency)
- **Warm query P95**: 52s (balanced), 35s (low_latency)
- **Memory footprint**: ~6GB (4GB model + 2GB embeddings + ChromaDB)
- **Storage**: 500MB for 2,126 chunks + embeddings
- **Concurrency**: FastAPI async supports 10+ parallel requests

**Bottlenecks:**
1. **Synthesis** (60-70% of total latency) - LLM generation is slowest
2. **Retrieval** (20-25%) - Vector search + BM25 scoring
3. **Planning** (8-12%) - LLM call for sub-query generation

---

## Maintenance & Evolution

### Adding New Nodes
1. Define node function in `app/graph/nodes.py`
2. Add to graph builder in `app/graph/workflow.py`
3. Update NavigatorState schema in `app/graph/state.py` if needed
4. Add trace logging and timing wrappers
5. Write integration tests in `backend/tests/`

### Tuning Retrieval
- Adjust weights in `app/config.py`: `HYBRID_WEIGHT_VECTOR`, `BM25_WEIGHT`
- Modify thresholds: `RETRIEVAL_MIN_SCORE`, `RETRIEVAL_MIN_CHUNKS`
- Update special boosting in `app/services/vector_store.py`: `_workflow_boost()`

### Adding Policy Patterns
- Edit `POLICY_PATTERNS` list in `app/services/policy.py`
- Add regex patterns for new adversarial/private query types
- Call `_reload_patterns()` to invalidate LRU cache

### Evaluation Workflow
1. Ingest new sources: `make ingest-pack`
2. Generate candidates: `make eval-candidates`
3. Manually curate gold labels in `backend/eval/dataset.jsonl`
4. Split dataset: `make eval-split`
5. Run eval: `make eval-dev`
6. Analyze report: `backend/eval/eval_report.{json,md}`

---

**Document Revision History:**
- 2026-03-29: Initial comprehensive architecture documentation
- Previous: High-level workflow notes only
