from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.config import AppConfig
from src.graph.nodes import GraphNodes, correction_decision, route_decision
from src.graph.state import GraphState


def build_graph(nodes: GraphNodes):
    graph = StateGraph(dict)

    graph.add_node("router", nodes.router)
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("reranker", nodes.reranker_node)
    graph.add_node("generator", nodes.generator)
    graph.add_node("self_correction", nodes.self_correction)
    graph.add_node("output", nodes.output)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retriever": "retriever", "output": "output"},
    )
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "self_correction")
    graph.add_conditional_edges(
        "self_correction",
        correction_decision,
        {"retriever": "retriever", "output": "output"},
    )
    graph.add_edge("output", END)

    return graph.compile()


class RAGService:
    def __init__(self, *, app_config: AppConfig, nodes: GraphNodes):
        self.app_config = app_config
        self.graph = build_graph(nodes)

    def run(self, query: str, request_id: str | None = None) -> GraphState:
        initial = GraphState(
            request_id=request_id or GraphState(query=query).request_id,
            query=query,
            max_iterations=self.app_config.max_iterations,
        )
        result = self.graph.invoke(initial.to_dict())
        return GraphState.model_validate(result)
