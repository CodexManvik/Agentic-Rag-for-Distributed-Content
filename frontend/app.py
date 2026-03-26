import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
HEALTH_URL = os.getenv("BACKEND_HEALTH_URL", "http://localhost:8000/health")

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


def _health_check() -> tuple[str, str]:
    try:
        health = requests.get(HEALTH_URL, timeout=5)
        if health.ok:
            payload = health.json()
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
    for event in trace:
        node = str(event.get("node", "unknown"))
        status = str(event.get("status", "unknown"))
        detail = str(event.get("detail", ""))
        icon = "✅" if status == "ok" else ("⚠️" if status == "weak" else "❌")
        st.write(f"{icon} {node} | {status} | {detail}")


def _render_citations(citations: list[dict[str, Any]], key_prefix: str) -> None:
    if not citations:
        st.caption("No citations returned")
        return
    for idx, citation in enumerate(citations, start=1):
        source = citation.get("source") or "unknown"
        url = citation.get("url") or ""
        snippet = citation.get("snippet") or ""
        source_type = citation.get("source_type") or "unknown"
        section = citation.get("section") or "n/a"
        page = citation.get("page_number")
        st.markdown(f"[{citation['index']}] {source}")
        st.caption(f"Type: {source_type} | Section: {section} | Page: {page if page else 'n/a'}")
        if url:
            st.link_button("Open source", url, key=f"{key_prefix}-cite-{idx}")
        if snippet:
            st.caption(snippet)

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

with st.sidebar:
    st.header("Demo Control Center")
    status, reason = _health_check()
    if status == "ok":
        st.success(f"Backend: {status}")
    elif status == "degraded":
        st.warning(f"Backend: {status} ({reason})")
    else:
        st.error(f"Backend: {status} ({reason})")

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

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.prefill = ""
        st.session_state.last_latency_ms = None

top_a, top_b, top_c = st.columns([1, 1, 1])
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            confidence = float(message.get("confidence", 0.0))
            rq = message.get("retrieval_quality", {})
            adequate = bool(rq.get("adequate", False)) if rq else False
            quality_class = "quality-ok" if adequate else "quality-weak"
            st.markdown(
                f"<div class='assistant-card'><div><span class='{quality_class}'>"
                f"Evidence {'adequate' if adequate else 'weak'}</span></div>"
                f"<div class='muted'>Confidence: {confidence:.2f} | "
                f"Max score: {float(rq.get('max_score', 0.0)):.2f} | "
                f"Source diversity: {int(rq.get('source_diversity', 0))} | "
                f"Chunks: {int(rq.get('chunk_count', 0))}</div></div>",
                unsafe_allow_html=True,
            )
            if message.get("abstained"):
                st.error(message.get("abstain_reason") or "System abstained due to insufficient evidence")
            with st.expander("Reasoning trace", expanded=False):
                _render_trace(message.get("trace", []))
            with st.expander("Citations", expanded=False):
                _render_citations(message.get("citations", []), key_prefix=f"history-{id(message)}")

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
                    response = requests.post(BACKEND_URL, json={"query": user_text}, timeout=90)
                except Exception as exc:
                    status.update(label="Workflow failed", state="error", expanded=True)
                    st.error(f"Backend unreachable: {exc}")
                    response = None

                if response is not None and response.ok:
                    data: dict[str, Any] = response.json()
                    answer = str(data.get("answer", ""))
                    citations = data.get("citations", [])
                    confidence = float(data.get("confidence", 0.0))
                    abstained = bool(data.get("abstained", False))
                    abstain_reason = data.get("abstain_reason")
                    trace = data.get("trace", [])
                    retrieval_quality = data.get("retrieval_quality", {})
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    st.session_state.last_latency_ms = elapsed_ms

                    status.write("Retriever: evidence fetched")
                    status.write("Synthesis: answer grounded in citations")
                    status.write("Validator: citation checks complete")
                    status.update(label=f"Workflow complete ({elapsed_ms/1000:.1f}s)", state="complete", expanded=False)

                    st.markdown(answer)
                    st.metric("Confidence", f"{confidence:.2f}")
                    st.caption(f"End-to-end latency: {elapsed_ms:.0f} ms")

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

                    with st.expander("Reasoning trace", expanded=False):
                        _render_trace(trace)

                    with st.expander("Citations", expanded=False):
                        _render_citations(citations, key_prefix=f"live-{len(st.session_state.messages)}")

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
                        }
                    )
                elif response is not None:
                    status.update(label="Workflow failed", state="error", expanded=True)
                    st.error(f"Backend error: {response.status_code} {response.text}")
