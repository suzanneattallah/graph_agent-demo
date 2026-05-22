"""
GraphAgent -- a DSPy ReAct agent positioned on a Neo4j knowledge graph.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import dspy
import mlflow
from neo4j import Driver

NAV_STATE_FILE = Path(r"C:\Projet\graph_agent\nav_state.json")


def _write_nav_state(state) -> None:
    NAV_STATE_FILE.write_text(
        json.dumps({"current":    state.current_node_id,
                    "visited":    state.visited,
                    "notes":      state.notes,
                    "moves":      state.moves,
                    "tool_calls": state.tool_calls}),
        encoding="utf-8",
    )


from .tools import (
    AgentState, get_driver,
    read_node, read_neighbours, read_incoming, read_outgoing,
    move_to, read_source_code, get_parent_class,
    find_path, search_node, get_call_chain, get_callers,
    history, add_note,
)


# -- Tool wrappers -------------------------------------------------------------
# Plain callables with docstrings -- DSPy ReAct picks them up automatically.
# DSPy autolog (log_traces=True) traces every call; no manual spans needed.

def _make_tools(state: AgentState, driver: Driver) -> list:

    def tool_read_node() -> str:
        """Read all properties of the node where the agent currently stands."""
        state.log_tool_call("tool_read_node")
        _write_nav_state(state)
        return json.dumps(read_node(state, driver), indent=2)

    def tool_read_neighbours(relation: str = "") -> str:
        """
        List all nodes directly connected to the current node.
        Optionally pass a relation type: CALLS, METHOD, CONTAINS, EXTENDS, IMPLEMENTS, IMPORTS.
        """
        state.log_tool_call("tool_read_neighbours", {"relation": relation})
        _write_nav_state(state)
        return json.dumps(read_neighbours(state, driver, relation or None), indent=2)

    def tool_read_incoming(relation: str = "") -> str:
        """List nodes that have an edge pointing TO the current node. Optional relation filter."""
        state.log_tool_call("tool_read_incoming", {"relation": relation})
        _write_nav_state(state)
        return json.dumps(read_incoming(state, driver, relation or None), indent=2)

    def tool_read_outgoing(relation: str = "") -> str:
        """List nodes that the current node points TO. Optional relation filter."""
        state.log_tool_call("tool_read_outgoing", {"relation": relation})
        _write_nav_state(state)
        return json.dumps(read_outgoing(state, driver, relation or None), indent=2)

    def tool_move_to(node_id: str) -> str:
        """
        Move the agent to another node by its exact id.
        Use search_node first if you do not know the exact id.
        """
        result = move_to(state, driver, node_id, via="tool_move_to")
        state.log_tool_call("tool_move_to", {"node_id": node_id})
        _write_nav_state(state)
        return json.dumps(result, indent=2)

    def tool_read_source_code(node_id: str = "") -> str:
        """
        Read the Java source code of the file linked to the current node.
        Optionally pass a node_id to move to that node first before reading.
        """
        if node_id:
            move_to(state, driver, node_id, via="tool_read_source_code")
        state.log_tool_call("tool_read_source_code", {"node_id": node_id})
        _write_nav_state(state)
        return json.dumps(read_source_code(state, driver), indent=2)

    def tool_get_parent_class() -> str:
        """If the current node is a Method, return its parent Class node."""
        state.log_tool_call("tool_get_parent_class")
        _write_nav_state(state)
        return json.dumps(get_parent_class(state, driver), indent=2)

    def tool_find_path(target_id: str) -> str:
        """Find the shortest path between the current node and target_id."""
        state.log_tool_call("tool_find_path", {"target_id": target_id})
        _write_nav_state(state)
        return json.dumps(find_path(state, driver, target_id), indent=2)

    def tool_search_node(name: str) -> str:
        """
        Search for nodes whose label or id contains the given string (case-insensitive).
        Returns up to 20 matches with their ids. Always use this before move_to.
        """
        state.log_tool_call("tool_search_node", {"name": name})
        _write_nav_state(state)
        return json.dumps(search_node(state, driver, name), indent=2)

    def tool_get_call_chain() -> str:
        """Trace all methods the current node calls recursively (up to 5 hops)."""
        state.log_tool_call("tool_get_call_chain")
        _write_nav_state(state)
        return json.dumps(get_call_chain(state, driver), indent=2)

    def tool_get_callers() -> str:
        """Find all methods that call the current node (up to 3 hops upstream)."""
        state.log_tool_call("tool_get_callers")
        _write_nav_state(state)
        return json.dumps(get_callers(state, driver), indent=2)

    def tool_history() -> str:
        """Return current position, visited nodes and accumulated notes."""
        state.log_tool_call("tool_history")
        _write_nav_state(state)
        return json.dumps(history(state, driver), indent=2)

    def tool_add_note(note: str) -> str:
        """Save an observation or conclusion to the agent memory."""
        result = add_note(state, driver, note)
        state.log_tool_call("tool_add_note", {"note": note[:80]})
        _write_nav_state(state)
        return json.dumps(result, indent=2)

    return [
        tool_read_node, tool_read_neighbours, tool_read_incoming, tool_read_outgoing,
        tool_move_to, tool_read_source_code, tool_get_parent_class, tool_find_path,
        tool_search_node, tool_get_call_chain, tool_get_callers, tool_history, tool_add_note,
    ]


# -- Signature -----------------------------------------------------------------

class GraphExploreSignature(dspy.Signature):
    """
    You are an expert code analyst navigating a Neo4j knowledge graph of the macrozheng/mall Java Spring Boot e-commerce codebase.
    You MUST explore the graph exclusively by calling tools — do NOT answer from prior knowledge or training data.

    MANDATORY RULES (violating any rule = invalid answer):
    1. Your VERY FIRST action MUST be tool_search_node() to locate the precise HTTP entry point (e.g., "ProductController", "SearchAPI", "@GetMapping", or route paths) before any other navigation.
    2. After locating or reading any node, ALWAYS call tool_read_neighbours() or tool_read_outgoing() to map its connections and identify the next architectural layer before proceeding.
    3. You MUST call tool_move_to() at least 5 times to visit at least 5 DIFFERENT nodes across the graph, ensuring progressive descent through the architecture.
    4. do NOT answer from prior knowledge — rely strictly on tool outputs, graph traversal results, and verified source code snippets.
    5. Trace execution flows strictly layer-by-layer: follow CALLS and METHOD edges from Controllers to Services, then to Mappers/Repositories/Elasticsearch clients. Explicitly justify each architectural transition in your reasoning.
    6. Use tool_read_source_code() on at least one key node per architectural layer to verify actual implementation, confirm method signatures, and validate data flow before moving deeper.
    7. Follow EXTENDS and IMPLEMENTS edges to trace class hierarchies and framework integrations when resolving interface implementations or base class behaviors.
    8. Maintain a strict exploration log in your reasoning: record every tool call, node ID visited, edge type traversed, and architectural layer crossed. This log is mandatory for final validation.
    9. Only write your final answer after visiting at least 5 distinct nodes and completing a full vertical trace covering a minimum of 3 architectural layers (Controller → Service → Data/ES).
    10. Cite specific node IDs, method signatures, and edge types discovered through tools in your final answer. Do not output raw tool names or JSON; provide a human-readable architectural narrative.
    11. If a path is blocked or ambiguous, backtrack using tool_read_neighbours() or refine your tool_search_node() query rather than guessing. Always prioritize graph-verified paths over assumptions.
    12. Your final answer MUST be a comprehensive, structured textual trace of the complete flow. Explicitly map each layer transition, cite visited node IDs, and explain the purpose of each component in the search pipeline.
    13. Validate your trace before concluding: ensure every claimed connection is backed by a tool output, and that the flow logically progresses from HTTP request handling to business logic delegation to data retrieval.
    14. When descending to the data layer, explicitly verify the retrieval mechanism (e.g., MyBatis Mapper, Elasticsearch Client, or Repository) by reading its source code and confirming the query/index structure.
    15. Never skip layers or assume implicit connections. Every hop between architectural layers must be explicitly triggered by a tool call and documented in your reasoning trace.
    """
    question: str   = dspy.InputField(desc="The question to answer about the codebase")
    start_node: str = dspy.InputField(desc="The node id where the agent currently stands")
    answer: str     = dspy.OutputField(
        desc="Detailed answer citing the exact node IDs and method names you visited"
    )


# -- GraphAgent ----------------------------------------------------------------

class GraphAgent:

    def __init__(self,
                 model: str = "qwen2.5-coder:7b",
                 api_base: Optional[str] = None,
                 mlflow_uri: str = "http://localhost:5000",
                 experiment: str = "graph-agent"):

        _OLLAMA_DEFAULT = "http://localhost:11434"
        if api_base and "11434" not in api_base:
            # OpenAI-compatible remote endpoint
            lm_model  = f"openai/{model}"
            lm_base   = api_base
            lm_key    = "none"
        else:
            # Ollama local
            lm_model  = f"ollama_chat/{model}"
            lm_base   = api_base or _OLLAMA_DEFAULT
            lm_key    = "ollama"

        self.lm = dspy.LM(lm_model, api_base=lm_base, api_key=lm_key)
        self.model = model
        dspy.configure(lm=self.lm)
        self.driver = get_driver()

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        try:
            mlflow.dspy.autolog(
                log_compiles=False,
                log_evals=False,
                log_traces=True,   # Full trace tree: LLM calls + Tool calls in one view
                silent=True,
            )
        except Exception:
            pass

    def run(self, question: str, start_node: Optional[str] = None,
            max_iters: int = 15) -> dict[str, Any]:

        state = AgentState(current_node_id=start_node)
        _write_nav_state(state)
        tools = _make_tools(state, self.driver)
        react = dspy.ReAct(GraphExploreSignature, tools=tools, max_iters=max_iters)

        node_label = start_node or "FREE"
        run_name = f"{node_label[:30]} -- {question[:40]}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "start_node": start_node or "NONE",
                "model":      self.model,
                "max_iters":  max_iters,
                "question":   question[:250],
            })

            print("Thinking...", end="", flush=True)
            t0 = time.perf_counter()
            result = react(question=question, start_node=start_node or "NONE")
            elapsed = time.perf_counter() - t0
            print(" done")

            mlflow.log_metrics({
                "elapsed_seconds": round(elapsed, 2),
                "n_visited_nodes": len(state.visited),
                "n_notes":         len(state.notes),
                "answer_chars":    len(result.answer),
            })
            mlflow.log_text(result.answer,                               "answer.txt")
            mlflow.log_text("\n".join(state.visited or ["--"]),          "visited_nodes.txt")
            if state.notes:
                mlflow.log_text("\n".join(state.notes),                  "notes.txt")
            all_nodes = ([start_node] if start_node else []) + state.visited
            nav_path = " -> ".join(all_nodes) if all_nodes else "--"
            mlflow.log_text(nav_path,                                    "navigation_path.txt")

            print(f"Run: http://localhost:5000/#/experiments/"
                  f"{run.info.experiment_id}/runs/{run.info.run_id}")

        return {
            "answer":  result.answer,
            "state":   state,
            "visited": state.visited,
            "notes":   state.notes,
        }

    def close(self):
        self.driver.close()