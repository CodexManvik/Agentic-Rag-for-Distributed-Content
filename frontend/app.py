import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="Smart Knowledge Navigator", layout="wide")
st.title("Smart Knowledge Navigator")
query = st.text_area("Ask a question", height=120, placeholder="What should I know about project onboarding?")

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Running multi-agent workflow..."):
            response = requests.post(BACKEND_URL, json={"query": query}, timeout=60)
        if response.ok:
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sub-queries")
            st.write(data["sub_queries"])
            st.subheader("Citations")
            for citation in data["citations"]:
                source = citation.get("source") or "unknown"
                url = citation.get("url") or ""
                snippet = citation.get("snippet") or ""
                st.markdown(f"[{citation['index']}] {source}")
                if url:
                    st.markdown(url)
                st.caption(snippet)
        else:
            st.error(f"Backend error: {response.status_code} {response.text}")
