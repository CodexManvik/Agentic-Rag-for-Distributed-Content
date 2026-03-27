import requests
import json
url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen3.5:0.8b",
    "prompt": "What is RAG?",
    "stream": False
}
res = requests.post(url, json=payload)
print("NO OPTIONS:", repr(res.json().get("response")))

payload["options"] = {"temperature": 0.1, "num_predict": 700}
res = requests.post(url, json=payload)
print("WITH OPTIONS:", repr(res.json().get("response")))

payload["options"] = {"temperature": 0.1, "num_predict": 700, "presence_penalty": 0.0, "repeat_penalty": 1.05}
res = requests.post(url, json=payload)
print("WITH PENALTIES:", repr(res.json().get("response")))
