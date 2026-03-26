import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
HEALTH_URL = os.getenv("BACKEND_HEALTH_URL", "http://localhost:8000/health")

st.set_page_config(page_title="Smart Knowledge Navigator", layout="wide")
st.title("Smart Knowledge Navigator")
st.caption("Public-source-only, local-model Agentic RAG")
st.warning("Ingest only public or approved content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

sample_queries = [
    "What are the key concepts behind LangChain RAG pipelines?",
    "How does LangGraph differ from a simple chain for workflow control?",
    "Summarize retrieval best practices from the docs with cited evidence.",
]
hard_queries = [
    "Compare orchestration patterns across LangChain and LangGraph docs and cite each claim.",
    "What governance and observability guidance is documented for production agent workflows?",
]

left, right = st.columns(2)
with left:
    st.subheader("Sample queries")
    for q in sample_queries:
        if st.button(q, key=f"sample-{q}"):
            st.session_state.prefill = q
with right:
    st.subheader("Hard queries")
    for q in hard_queries:
        if st.button(q, key=f"hard-{q}"):
            st.session_state.prefill = q

with st.expander("Service status", expanded=False):
    try:
        health = requests.get(HEALTH_URL, timeout=5)
        if health.ok:
            st.success("Backend is reachable")
        else:
            st.error(f"Backend returned {health.status_code}")
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("confidence") is not None:
            confidence = float(message["confidence"])
            st.metric("Confidence", f"{confidence:.2f}")
        if message.get("retrieval_quality"):
            rq = message["retrieval_quality"]
            st.caption(
                f"Retrieval quality: adequate={rq.get('adequate')} max={rq.get('max_score', 0):.2f} "
                f"diversity={rq.get('source_diversity', 0)} chunks={rq.get('chunk_count', 0)}"
            )
        if message.get("trace"):
            with st.expander("Agent trace", expanded=False):
                for event in message["trace"]:
                    st.write(f"{event.get('node')}: {event.get('status')} - {event.get('detail')}")
        if message.get("abstained"):
            st.error(message.get("abstain_reason") or "System abstained due to insufficient evidence")
        if message.get("citations"):
            with st.expander("Citations", expanded=False):
                for citation in message["citations"]:
                    source = citation.get("source") or "unknown"
                    url = citation.get("url") or ""
                    snippet = citation.get("snippet") or ""
                    source_type = citation.get("source_type") or "unknown"
                    section = citation.get("section") or "n/a"
                    page = citation.get("page_number")
                    st.markdown(f"[{citation['index']}] {source}")
                    st.caption(f"Type: {source_type} | Section: {section} | Page: {page if page else 'n/a'}")
                    if url:
                        st.link_button("Open source", url)
                    st.caption(snippet)

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
            with st.status("Running agent workflow", expanded=True) as status:
                status.write("Planning Agent: Analyzing intent...")
                time.sleep(0.5)
                status.write("Retrieval Agent: Scanning vector store...")
                time.sleep(0.5)
                status.write("Adequacy Check: Scoring evidence quality...")
                time.sleep(0.5)
                status.write("Synthesis Agent: Building cited response...")
                time.sleep(0.5)
                status.write("Citation Validator: Verifying grounding...")
                time.sleep(0.5)

                response = requests.post(BACKEND_URL, json={"query": user_text}, timeout=60)
                if response.ok:
                    data: dict[str, Any] = response.json()
                    answer = str(data.get("answer", ""))
                    citations = data.get("citations", [])
                    confidence = float(data.get("confidence", 0.0))
                    abstained = bool(data.get("abstained", False))
                    abstain_reason = data.get("abstain_reason")
                    trace = data.get("trace", [])
                    retrieval_quality = data.get("retrieval_quality", {})

                    status.update(label="Workflow complete", state="complete", expanded=False)
                    st.metric("Confidence", f"{confidence:.2f}")
                    if abstained:
                        st.error(abstain_reason or "System abstained due to insufficient evidence")
                    st.markdown(answer)
                    if retrieval_quality:
                        st.caption(
                            f"Retrieval quality: adequate={retrieval_quality.get('adequate')} "
                            f"max={retrieval_quality.get('max_score', 0):.2f} "
                            f"diversity={retrieval_quality.get('source_diversity', 0)} "
                            f"chunks={retrieval_quality.get('chunk_count', 0)}"
                        )
                    if trace:
                        with st.expander("Agent trace", expanded=False):
                            for event in trace:
                                st.write(
                                    f"{event.get('node')}: {event.get('status')} - {event.get('detail')}"
                                )
                    with st.expander("Citations", expanded=False):
                        for citation in citations:
                            source = citation.get("source") or "unknown"
                            url = citation.get("url") or ""
                            snippet = citation.get("snippet") or ""
                            source_type = citation.get("source_type") or "unknown"
                            section = citation.get("section") or "n/a"
                            page = citation.get("page_number")
                            st.markdown(f"[{citation['index']}] {source}")
                            st.caption(
                                f"Type: {source_type} | Section: {section} | Page: {page if page else 'n/a'}"
                            )
                            if url:
                                st.link_button("Open source", url)
                            st.caption(snippet)

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
                else:
                    status.update(label="Workflow failed", state="error", expanded=True)
                    st.error(f"Backend error: {response.status_code} {response.text}")
