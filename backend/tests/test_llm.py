import requests
import json
url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen3.5:0.8b",
    "prompt": "You are a knowledge synthesis agent. Answer the USER QUESTION using ONLY the information in the CONTEXT below.\n\nCONTEXT (each section is prefixed with its chunk number):\n[1] RAG is a technique.\n\nINSTRUCTIONS:\n- Base your answer strictly on the CONTEXT. Do not use outside knowledge.\n- Write a concise answer.\n- Set confidence to a float between 0.0 and 1.0 (e.g. 0.85 if well-supported).\n- Set abstain_reason to null unless you genuinely cannot answer from the context.\n- cited_indices is a JSON array of chunk numbers you referenced, e.g. [1, 2].\n\nOUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no extra text:\n{\"answer\": \"<your answer>\", \"cited_indices\": [1], \"confidence\": 0.8, \"abstain_reason\": null}\n\nUSER QUESTION: What is RAG?\n",
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 700}
}
try:
    res = requests.post(url, json=payload)
    print("STATUS:", res.status_code)
    print("RESPONSE:", repr(res.json().get("response")))
except Exception as e:
    print("ERROR:", e)
