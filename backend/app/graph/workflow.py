from langgraph.graph import END, StateGraph

from app.graph.nodes import planning_agent, retrieval_agent, synthesis_agent
from app.graph.state import NavigatorState


def build_graph():
    graph = StateGraph(NavigatorState)
    graph.add_node("planning", planning_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.set_entry_point("planning")
    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


workflow = build_graph()


def run_workflow(query: str) -> NavigatorState:
    initial_state: NavigatorState = {
        "original_query": query,
        "sub_queries": [],
        "retrieved_chunks": [],
        "final_response": "",
        "citations": [],
    }
    return workflow.invoke(initial_state)
