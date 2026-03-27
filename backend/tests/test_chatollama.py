from app.graph.nodes import _synthesis_prompt
from app.services.llm import get_chat_model
from langchain_core.messages import HumanMessage
p = _synthesis_prompt({"original_query": "What is RAG?"}, "[1] RAG is...", False)
m = get_chat_model(700)
res = m.invoke(p)
print(repr(res.content))
