import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="Smart Knowledge Navigator", layout="wide")
st.title("Smart Knowledge Navigator")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Citations", expanded=False):
                for citation in message["citations"]:
                    source = citation.get("source") or "unknown"
                    url = citation.get("url") or ""
                    snippet = citation.get("snippet") or ""
                    st.markdown(f"[{citation['index']}] {source}")
                    if url:
                        st.markdown(url)
                    st.caption(snippet)

prompt = st.chat_input("Ask a question")
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
                status.write("Synthesis Agent: Building cited response...")
                time.sleep(0.5)

                response = requests.post(BACKEND_URL, json={"query": user_text}, timeout=60)
                if response.ok:
                    data: dict[str, Any] = response.json()
                    answer = str(data.get("answer", ""))
                    citations = data.get("citations", [])

                    status.update(label="Workflow complete", state="complete", expanded=False)
                    st.markdown(answer)
                    with st.expander("Citations", expanded=False):
                        for citation in citations:
                            source = citation.get("source") or "unknown"
                            url = citation.get("url") or ""
                            snippet = citation.get("snippet") or ""
                            st.markdown(f"[{citation['index']}] {source}")
                            if url:
                                st.markdown(url)
                            st.caption(snippet)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        }
                    )
                else:
                    status.update(label="Workflow failed", state="error", expanded=True)
                    st.error(f"Backend error: {response.status_code} {response.text}")
