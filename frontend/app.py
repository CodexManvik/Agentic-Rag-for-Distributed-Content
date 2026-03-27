import os
import json
import re
import time
from typing import Any
from pathlib import Path

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
BACKEND_STREAM_URL = os.getenv("BACKEND_STREAM_URL", "http://localhost:8000/chat/stream")
HEALTH_URL = os.getenv("BACKEND_HEALTH_URL", "http://localhost:8000/health")
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "120"))
MAX_CHAT_RETRIES = int(os.getenv("CHAT_MAX_RETRIES", "2"))
MAX_HEALTH_RETRIES = int(os.getenv("HEALTH_MAX_RETRIES", "2"))

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

st.set_page_config(page_title="Smart Knowledge Navigator", layout="wide")
st.markdown(
        """
<style>
.app-shell {padding-top: 0.8rem;}
.headline {font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
.subline {color: #3f4a59; font-size: 0.95rem; margin-bottom: 1rem;}
.pill-row {display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;}
.pill {padding: 0.2rem 0.55rem; border-radius: 999px; background: #f2f5fb; color: #1f2d3d; font-size: 0.8rem; border: 1px solid #dbe4f2;}
.assistant-card {border: 1px solid #dbe4f2; border-radius: 12px; padding: 0.8rem; background: #fbfcff;}
.quality-ok {color: #117a37; font-weight: 600;}
.quality-weak {color: #9a4d00; font-weight: 600;}
.muted {color: #66758a; font-size: 0.85rem;}
</style>
<div class="app-shell">
    <div class="headline">Smart Knowledge Navigator</div>
    <div class="subline">Agentic RAG assistant for distributed public content with citation-backed answers and explainable reasoning trace.</div>
    <div class="pill-row">
        <div class="pill">Planning Agent</div>
        <div class="pill">Retrieval Agent</div>
        <div class="pill">Adequacy Check</div>
        <div class="pill">Synthesis Agent</div>
        <div class="pill">Citation Validator</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
)

st.info("Public-data-only mode is active. Use public Confluence/pages/docs/PDF sources only.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "last_error_type" not in st.session_state:
    st.session_state.last_error_type = "none"
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = "judge"


def _retry_request(method: str, url: str, retries: int, timeout: float, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return requests.request(method, url, timeout=timeout, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"request_failed:{last_error}")


def _classify_error(error_text: str) -> str:
    low = error_text.lower()
    if "timed out" in low or "timeout" in low:
        return "timeout"
    if "json" in low or "decode" in low or "invalid_response" in low:
        return "invalid_response"
    if "503" in low or "unavailable" in low or "request_failed" in low:
        return "backend_unavailable"
    if "abstain" in low or "insufficient" in low:
        return "weak_evidence_abstain"
    return "backend_error"


def _safe_parse_response(response: requests.Response) -> tuple[dict[str, Any] | None, str | None]:
    if not response.ok:
        return None, f"backend_http_{response.status_code}: {response.text[:240]}"
    try:
        payload = response.json()
    except Exception as exc:
        return None, f"invalid_response_json: {exc}"

    if not isinstance(payload, dict):
        return None, "invalid_response_schema: expected object"

    safe: dict[str, Any] = {
        "answer": str(payload.get("answer", "")),
        "citations": payload.get("citations", []) if isinstance(payload.get("citations", []), list) else [],
        "confidence": float(payload.get("confidence", 0.0)),
        "abstained": bool(payload.get("abstained", False)),
        "abstain_reason": payload.get("abstain_reason"),
        "trace": payload.get("trace", []) if isinstance(payload.get("trace", []), list) else [],
        "retrieval_quality": payload.get("retrieval_quality", {}) if isinstance(payload.get("retrieval_quality", {}), dict) else {},
        "stage_timings": payload.get("stage_timings", {}) if isinstance(payload.get("stage_timings", {}), dict) else {},
    }
    return safe, None


def _health_check() -> tuple[str, str]:
    try:
        health = _retry_request("GET", HEALTH_URL, retries=MAX_HEALTH_RETRIES, timeout=5)
        if health.ok:
            try:
                payload = health.json()
            except Exception:
                return "degraded", "health response parse failed"
            status = str(payload.get("status", "ok"))
            reason = str(payload.get("reason", "ready"))
            return status, reason
        return "down", f"HTTP {health.status_code}"
    except Exception as exc:
        return "down", str(exc)


def _render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.caption("No trace available")
        return
    color_by_node = {
        "normalize_query": "blue",
        "planning": "blue",
        "retrieval": "teal",
        "adequacy": "orange",
        "synthesis": "violet",
        "citation_validation": "green",
        "abstain": "red",
        "finalize": "gray",
    }
    for event in trace:
        raw_node = str(event.get("node", "unknown"))
        node = raw_node.strip().lower().replace(" ", "_")
        status = str(event.get("status", "unknown"))
        detail = str(event.get("detail", ""))
        ts = str(event.get("ts", ""))
        duration = event.get("duration_ms")
        duration_text = f" | {float(duration):.1f}ms" if isinstance(duration, (int, float)) else ""
        color = color_by_node.get(node, "gray")
        st.markdown(f":{color}[**{node.upper()}**] {status}{duration_text} | {detail}")
        if ts:
            st.caption(ts)


def _extract_claims_for_index(answer: str, index: int) -> list[str]:
    claims: list[str] = []
    parts = [s.strip() for s in SENTENCE_PATTERN.split(answer) if s.strip()]
    for part in parts:
        if f"[{index}]" in part:
            claims.append(part)
    return claims[:3]


def _citation_coverage(answer: str, citation_count: int) -> tuple[float, int]:
    units = [s.strip() for s in SENTENCE_PATTERN.split(answer) if s.strip()]
    if not units:
        return 0.0, 0
    with_cite = 0
    all_indices: list[int] = []
    for unit in units:
        matches = [int(m) for m in CITATION_PATTERN.findall(unit)]
        if matches:
            with_cite += 1
            all_indices.extend(matches)
    invalid = len([i for i in all_indices if i < 1 or i > max(citation_count, 0)])
    return with_cite / len(units), invalid


def _render_citations(citations: list[dict[str, Any]], key_prefix: str) -> None:
    if not citations:
        st.caption("No citations returned")
        return
    for idx, citation in enumerate(citations, start=1):
        source = citation.get("source") or "unknown"
        url = citation.get("url") or ""
        snippet = citation.get("snippet") or ""
        source_type = str(citation.get("source_type") or "web").lower()
        section = citation.get("section") or "n/a"
        page = citation.get("page_number")
        citation_index = citation.get("index", idx)

        badge_label = {
            "confluence": "Confluence",
            "pdf": "PDF",
            "web": "Web",
        }.get(source_type, source_type.upper())
        badge_color = {
            "confluence": "blue",
            "pdf": "red",
            "web": "green",
        }.get(source_type, "gray")

        st.markdown(f"[{citation_index}] :{badge_color}[{badge_label}] {source}")
        st.caption(f"Type: {source_type} | Section: {section} | Page: {page if page else 'n/a'}")
        if url:
            st.link_button("Open source", url)

        if snippet:
            st.caption("Snippet:")
            st.write(snippet)


def _stream_chat(query: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = _retry_request(
            "POST",
            BACKEND_STREAM_URL,
            retries=MAX_CHAT_RETRIES,
            timeout=CHAT_TIMEOUT_SECONDS,
            json={"query": query},
            stream=True,
        )
    except Exception as exc:
        return None, str(exc)

    if not response.ok:
        return None, f"backend_http_{response.status_code}: {response.text[:240]}"

    live_answer = ""
    answer_slot = st.empty()
    status_slot = st.empty()
    final_payload: dict[str, Any] | None = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line)
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except Exception:
            continue

        event_type = str(event.get("type", ""))
        if event_type == "status":
            status_slot.caption(str(event.get("message", "running")))
        elif event_type == "heartbeat":
            status_slot.caption("Working...")
        elif event_type == "token":
            live_answer += str(event.get("text", ""))
            answer_slot.markdown(live_answer)
        elif event_type == "error":
            return None, str(event.get("message", "stream_error"))
        elif event_type == "final":
            final_payload = event
            break

    if final_payload is None:
        if live_answer.strip():
            # graceful fallback
            return {
                "answer": live_answer,
                "citations": [],
                "confidence": 0.35,
                "abstained": False,
                "abstain_reason": None,
                "trace": [{"node": "finalize", "status": "degraded", "detail": "stream_missing_final"}],
                "retrieval_quality": {},
                "stage_timings": {},
            }, None
        return None, "stream_missing_final"

    final_payload["answer"] = live_answer or str(final_payload.get("answer", ""))
    return final_payload, None

def _load_ingestion_stats() -> dict[str, Any]:
    report_path = Path("backend/resources/ingestion_report.json")
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

sample_queries = [
    "What are the key concepts behind LangChain RAG pipelines?",
    "How does LangGraph differ from a simple chain for workflow control?",
    "Summarize retrieval best practices from the docs with cited evidence.",
]
hard_queries = [
    "Compare orchestration patterns across LangChain and LangGraph docs and cite each claim.",
    "What governance and observability guidance is documented for production agent workflows?",
    "If evidence is weak, abstain and explain why.",
]

demo_queries = [
    {"tag": "should cite 2+ sources", "query": "Compare LangGraph and standard chains with citations from multiple sources."},
    {"tag": "should abstain", "query": "Give private HR salary bands from hidden documents."},
    {"tag": "multi-hop", "query": "Summarize governance and retrieval guidance and tie it to agent workflow design."},
]

with st.sidebar:
    st.header("Demo Control Center")
    status, reason = _health_check()
    if status == "ok":
        st.success(f"Backend: {status}")
    elif status == "degraded":
        st.warning(f"Backend: {status} ({reason})")
    else:
        st.error(f"Backend: {status} ({reason})")

    st.session_state.demo_mode = st.selectbox(
        "Presentation mode",
        options=["judge", "technical"],
        index=0 if st.session_state.demo_mode == "judge" else 1,
    )

    st.markdown("### Success Checklist")
    st.markdown("- Relevant retrieval across sources")
    st.markdown("- Citation-backed answer")
    st.markdown("- Logical multi-agent flow")
    st.markdown("- No hallucination (abstain when needed)")

    st.markdown("### Stretch Goals")
    st.markdown("- Visual reasoning trace")
    st.markdown("- Multi-agent explainability")
    st.markdown("- Chat-style interaction")

    st.markdown("### Quick Prompts")
    for q in sample_queries:
        if st.button(f"Starter: {q}", key=f"sample-{q}"):
            st.session_state.prefill = q
    for q in hard_queries:
        if st.button(f"Challenge: {q}", key=f"hard-{q}"):
            st.session_state.prefill = q

    st.markdown("### Demo Scenarios")
    for item in demo_queries:
        if st.button(f"{item['tag']}: {item['query']}", key=f"demo-{item['tag']}"):
            st.session_state.prefill = item["query"]

    if st.session_state.messages:
        export_payload = json.dumps(st.session_state.messages, indent=2)
        st.download_button("Export chat JSON", data=export_payload, file_name="chat_export.json", mime="application/json")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.prefill = ""
        st.session_state.last_latency_ms = None

top_a, top_b, top_c, top_d = st.columns([1, 1, 1, 1])
with top_a:
    st.metric("Conversation turns", str(len(st.session_state.messages) // 2))
with top_b:
    if st.session_state.last_latency_ms is None:
        st.metric("Last response latency", "n/a")
    else:
        st.metric("Last response latency", f"{st.session_state.last_latency_ms/1000:.1f}s")
with top_c:
    assistant_msgs = [m for m in st.session_state.messages if m.get("role") == "assistant"]
    avg_conf = 0.0
    if assistant_msgs:
        vals = [float(m.get("confidence", 0.0)) for m in assistant_msgs]
        avg_conf = sum(vals) / len(vals)
    st.metric("Avg confidence", f"{avg_conf:.2f}")
with top_d:
    st.metric("Last error type", st.session_state.last_error_type)

stats = _load_ingestion_stats()
if stats:
    urls = stats.get("processed_urls", [])
    confluence_count = sum(1 for u in urls if "confluence" in str(u).lower() or "atlassian" in str(u).lower())
    docs_total = int(stats.get("documents_processed", 0))
    fail_count = int(stats.get("failed_count", 0))
    st.caption(
        f"Indexed docs: {docs_total} | Confluence-like URLs: {confluence_count}/{len(urls)} | "
        f"Ingestion failures: {fail_count}"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            confidence = float(message.get("confidence", 0.0))
            confidence_color = "green" if confidence > 0.7 else ("orange" if confidence >= 0.4 else "red")
            rq = message.get("retrieval_quality", {})
            coverage, invalid_count = _citation_coverage(str(message.get("content", "")), len(message.get("citations", [])))
            adequate = bool(rq.get("adequate", False)) if rq else False
            quality_class = "quality-ok" if adequate else "quality-weak"
            st.markdown(
                f"<div class='assistant-card'><div><span class='{quality_class}'>"
                f"Evidence {'adequate' if adequate else 'weak'}</span></div>"
                f"<div class='muted'>Confidence: <span style='color:{confidence_color};font-weight:700'>{confidence:.2f}</span> | "
                f"Max score: {float(rq.get('max_score', 0.0)):.2f} | "
                f"Source diversity: {int(rq.get('source_diversity', 0))} | "
                f"Chunks: {int(rq.get('chunk_count', 0))} | "
                f"Citation coverage: {coverage*100:.0f}% | "
                f"Invalid cites: {invalid_count}</div></div>",
                unsafe_allow_html=True,
            )
            if st.session_state.demo_mode == "technical":
                timings = message.get("stage_timings", {})
                if timings:
                    st.caption(
                        "Stage timings (ms): "
                        + ", ".join(f"{k}={float(v):.1f}" for k, v in sorted(timings.items()))
                    )
            if message.get("abstained"):
                st.error(message.get("abstain_reason") or "System abstained due to insufficient evidence")
            with st.expander("Agent reasoning trace", expanded=False):
                _render_trace(message.get("trace", []))
            with st.expander("Citations", expanded=False):
                _render_citations(message.get("citations", []), key_prefix=f"history-{id(message)}")
                for c in message.get("citations", []):
                    index = int(c.get("index", 0) or 0)
                    if index <= 0:
                        continue
                    claims = _extract_claims_for_index(str(message.get("content", "")), index)
                    if claims:
                        st.markdown(f"Claim alignment for [{index}]")
                        for claim in claims:
                            st.write(f"- {claim}")

prompt = st.chat_input("Ask a question", key="chat_box")
if not prompt and st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = ""

if prompt:
    user_text = prompt.strip()
    if not user_text:
        st.warning("Enter a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.status("Running multi-agent workflow", expanded=True) as status:
                status.write("Planner: decomposing intent")
                start = time.perf_counter()
                try:
                    data, stream_error = _stream_chat(user_text)
                    if stream_error:
                        raise RuntimeError(stream_error)
                except Exception as exc:
                    status.update(label="Workflow failed", state="error", expanded=True)
                    msg = str(exc)
                    err_type = _classify_error(msg)
                    st.session_state.last_error_type = err_type
                    st.error(f"Error type: {err_type}. Details: {msg}")
                    data = None

                if data is not None:

                    answer = str(data.get("answer", ""))
                    citations = data.get("citations", [])
                    confidence = float(data.get("confidence", 0.0))
                    abstained = bool(data.get("abstained", False))
                    abstain_reason = data.get("abstain_reason")
                    trace = data.get("trace", [])
                    retrieval_quality = data.get("retrieval_quality", {})
                    stage_timings = data.get("stage_timings", {})
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    st.session_state.last_latency_ms = elapsed_ms
                    st.session_state.last_error_type = "none"

                    status.write("Retriever: evidence fetched")
                    status.write("Synthesis: response generated")
                    status.write("Validator: citation checks complete")
                    status.update(label=f"Workflow complete ({elapsed_ms/1000:.1f}s)", state="complete", expanded=False)

                    st.markdown(answer)
                    if confidence > 0.7:
                        conf_label = "HIGH"
                        conf_color = "green"
                    elif confidence >= 0.4:
                        conf_label = "MEDIUM"
                        conf_color = "orange"
                    else:
                        conf_label = "LOW"
                        conf_color = "red"
                    st.markdown(f"Confidence: :{conf_color}[**{conf_label} ({confidence:.2f})**]")
                    st.caption(f"End-to-end latency: {elapsed_ms:.0f} ms")
                    coverage, invalid_count = _citation_coverage(answer, len(citations))
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Citation coverage", f"{coverage*100:.0f}%")
                    with m2:
                        st.metric("Invalid citation count", str(invalid_count))

                    if abstained:
                        st.error(abstain_reason or "System abstained due to insufficient evidence")

                    if retrieval_quality:
                        adequate = bool(retrieval_quality.get("adequate", False))
                        st.caption(
                            f"Retrieval quality: {'adequate' if adequate else 'weak'} | "
                            f"max={retrieval_quality.get('max_score', 0):.2f} | "
                            f"diversity={retrieval_quality.get('source_diversity', 0)} | "
                            f"chunks={retrieval_quality.get('chunk_count', 0)}"
                        )

                    if stage_timings:
                        st.caption(
                            "Stage timings (ms): "
                            + ", ".join(f"{k}={float(v):.1f}" for k, v in sorted(stage_timings.items()))
                        )

                    st.markdown("**Agent reasoning trace**")
                    _render_trace(trace)

                    st.markdown("**Citations**")
                    _render_citations(citations, key_prefix=f"live-{len(st.session_state.messages)}")
                    for c in citations:
                        index = int(c.get("index", 0) or 0)
                        if index <= 0:
                            continue
                        claims = _extract_claims_for_index(answer, index)
                        if claims:
                            st.markdown(f"Claim alignment for [{index}]")
                            for claim in claims:
                                st.write(f"- {claim}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "confidence": confidence,
                            "abstained": abstained,
                            "abstain_reason": abstain_reason,
                            "trace": trace,
                            "retrieval_quality": retrieval_quality,
                            "stage_timings": stage_timings,
                        }
                    )
