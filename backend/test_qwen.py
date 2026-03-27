import requests
url = "http://localhost:11434/api/chat"
payload = {
    "model": "qwen3.5:2b",
    "messages": [
        {"role": "user", "content": "What is agentic RAG?"}
    ],
    "stream": False
}
print(requests.post(url, json=payload).json())
