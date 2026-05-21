"""
Pre-built Cypher tools for the GraphAgent.

Every tool operates relative to an AgentState object so the LLM never
has to write raw Cypher — it just calls named functions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase, Driver


# ── Connection ────────────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "agent_recherche_neo4j"

def get_driver() -> Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Agent state ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    current_node_id: Optional[str] = None
    visited: list[str] = field(default_factory=list)
    notes: list[str]   = field(default_factory=list)
    moves: list[dict]  = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)

    def move(self, node_id: str, via: str = "move_to") -> None:
        if self.current_node_id is not None:
            self.moves.append({
                "from":      self.current_node_id,
                "to":        node_id,
                "via":       via,
                "timestamp": time.time(),
            })
            self.visited.append(self.current_node_id)
        self.current_node_id = node_id

    def log_tool_call(self, tool: str, args: dict | None = None) -> None:
        self.tool_calls.append({
            "tool":      tool,
            "node":      self.current_node_id,
            "args":      args or {},
            "timestamp": time.time(),
        })

    def add_note(self, note: str) -> None:
        self.notes.append(note)


# ── Helper ────────────────────────────────────────────────────────────────────

def _run(driver: Driver, query: str, **params) -> list[dict]:
    with driver.session(database="neo4j") as s:
        return [r.data() for r in s.run(query, **params)]


# ── Tools ─────────────────────────────────────────────────────────────────────

def read_node(state: AgentState, driver: Driver) -> dict:
    """Return all properties of the current node."""
    if state.current_node_id is None:
        return {"error": "No current node. Use search_node() to find a node, then move_to() it."}
    rows = _run(driver,
        "MATCH (n {id: $id}) RETURN n, labels(n) AS labels",
        id=state.current_node_id)
    if not rows:
        return {"error": f"Node '{state.current_node_id}' not found."}
    row = rows[0]
    props = dict(row["n"])
    props["_labels"] = row["labels"]
    return props


def read_neighbours(state: AgentState, driver: Driver,
                    relation: Optional[str] = None) -> list[dict]:
    """
    Return all directly connected nodes (both directions).
    Optionally filter by relation type (e.g. 'CALLS', 'METHOD', 'EXTENDS').
    """
    if state.current_node_id is None:
        return [{"error": "No current node. Use search_node() first, then move_to()."}]
    if relation:
        rel_clause = f"[r:{relation.upper()}]"
    else:
        rel_clause = "[r]"

    rows = _run(driver, f"""
        MATCH (n {{id: $id}})-{rel_clause}-(nb)
        RETURN nb.id AS id, nb.label AS label,
               labels(nb) AS node_labels,
               type(r) AS relation,
               startNode(r).id = $id AS outgoing
        ORDER BY relation, label
    """, id=state.current_node_id)
    return rows


def read_incoming(state: AgentState, driver: Driver,
                  relation: Optional[str] = None) -> list[dict]:
    """Return nodes that point TO the current node."""
    if state.current_node_id is None:
        return [{"error": "No current node. Use search_node() first, then move_to()."}]
    rel_clause = f"[r:{relation.upper()}]" if relation else "[r]"
    rows = _run(driver, f"""
        MATCH (src)-{rel_clause}->(n {{id: $id}})
        RETURN src.id AS id, src.label AS label,
               labels(src) AS node_labels, type(r) AS relation
        ORDER BY relation, label
    """, id=state.current_node_id)
    return rows


def read_outgoing(state: AgentState, driver: Driver,
                  relation: Optional[str] = None) -> list[dict]:
    """Return nodes the current node points TO."""
    if state.current_node_id is None:
        return [{"error": "No current node. Use search_node() first, then move_to()."}]
    rel_clause = f"[r:{relation.upper()}]" if relation else "[r]"
    rows = _run(driver, f"""
        MATCH (n {{id: $id}})-{rel_clause}->(tgt)
        RETURN tgt.id AS id, tgt.label AS label,
               labels(tgt) AS node_labels, type(r) AS relation
        ORDER BY relation, label
    """, id=state.current_node_id)
    return rows


def move_to(state: AgentState, driver: Driver, node_id: str, via: str = "move_to") -> dict:
    """Move agent to a different node. Returns the new node's properties."""
    rows = _run(driver, "MATCH (n {id: $id}) RETURN n", id=node_id)
    if not rows:
        return {"error": f"Node '{node_id}' not found. Use search_node() to find valid IDs."}
    state.move(node_id, via=via)
    from_node = state.visited[-1] if state.visited else None
    return {"moved_to": node_id, "from": from_node, "node": dict(rows[0]["n"])}


def read_source_code(state: AgentState, driver: Driver,
                     java_root: str = r"C:\Projet\mall") -> dict:
    """
    Read the Java source file associated with the current node.
    Works for File, Class and Method nodes.
    """
    props = read_node(state, driver)
    src = props.get("source_file", "")
    if not src:
        return {"error": "No source_file on this node."}

    full_path = Path(java_root) / src
    if not full_path.exists():
        return {"error": f"File not found on disk: {full_path}"}

    content = full_path.read_text(encoding="utf-8", errors="replace")
    loc = props.get("source_location", "")

    # If a line hint exists, return a focused excerpt (±10 lines)
    if loc and loc.startswith("L"):
        try:
            line_no = int(loc[1:])
            lines = content.splitlines()
            start = max(0, line_no - 10)
            end   = min(len(lines), line_no + 10)
            excerpt = "\n".join(
                f"{i+1:4d}  {l}" for i, l in enumerate(lines[start:end], start=start)
            )
            return {"file": src, "around_line": line_no,
                    "excerpt": excerpt, "total_lines": len(lines)}
        except ValueError:
            pass

    return {"file": src, "content": content}


def get_parent_class(state: AgentState, driver: Driver) -> dict:
    """If current node is a Method, return its parent Class."""
    if state.current_node_id is None:
        return {"error": "No current node. Use search_node() first, then move_to()."}
    rows = _run(driver, """
        MATCH (c:Class)-[:METHOD]->(n {id: $id})
        RETURN c.id AS id, c.label AS label, c.source_file AS source_file
    """, id=state.current_node_id)
    if not rows:
        rows = _run(driver, """
            MATCH (c)-[:CONTAINS]->(n {id: $id})
            RETURN c.id AS id, c.label AS label, c.source_file AS source_file
        """, id=state.current_node_id)
    if not rows:
        return {"error": "No parent class found for this node."}
    return rows[0]


def find_path(state: AgentState, driver: Driver, target_id: str) -> dict:
    """Find the shortest path between current node and a target node."""
    if state.current_node_id is None:
        return {"error": "No current node. Use search_node() first, then move_to()."}
    rows = _run(driver, """
        MATCH path = shortestPath(
            (a {id: $src})-[*..10]-(b {id: $tgt})
        )
        RETURN [n IN nodes(path) | {id: n.id, label: n.label}] AS nodes,
               [r IN relationships(path) | type(r)] AS relations,
               length(path) AS length
    """, src=state.current_node_id, tgt=target_id)
    if not rows:
        return {"error": f"No path found between '{state.current_node_id}' and '{target_id}'."}
    return rows[0]


def search_node(state: AgentState, driver: Driver, name: str) -> list[dict]:
    """Search nodes whose label contains the given string (case-insensitive)."""
    rows = _run(driver, """
        MATCH (n)
        WHERE toLower(n.label) CONTAINS toLower($name)
           OR toLower(n.id)    CONTAINS toLower($name)
        RETURN n.id AS id, n.label AS label,
               labels(n) AS node_labels, n.source_file AS source_file
        ORDER BY size(n.label)
        LIMIT 20
    """, name=name)
    return rows


def get_call_chain(state: AgentState, driver: Driver, depth: int = 5) -> list[dict]:
    """
    Trace the call chain FROM the current node up to `depth` hops.
    Shows which methods this node ultimately calls.
    """
    if state.current_node_id is None:
        return [{"error": "No current node. Use search_node() first, then move_to()."}]
    rows = _run(driver, f"""
        MATCH path = (n {{id: $id}})-[:CALLS*1..{depth}]->(m)
        RETURN [x IN nodes(path) | x.label] AS chain, length(path) AS depth
        ORDER BY depth
        LIMIT 30
    """, id=state.current_node_id)
    return rows


def get_callers(state: AgentState, driver: Driver, depth: int = 3) -> list[dict]:
    """Who calls the current node? Traces up the call chain."""
    if state.current_node_id is None:
        return [{"error": "No current node. Use search_node() first, then move_to()."}]
    rows = _run(driver, f"""
        MATCH path = (caller)-[:CALLS*1..{depth}]->(n {{id: $id}})
        RETURN [x IN nodes(path) | x.label] AS chain, length(path) AS depth
        ORDER BY depth
        LIMIT 20
    """, id=state.current_node_id)
    return rows


def history(state: AgentState, driver: Driver) -> dict:
    """Return the list of previously visited node IDs and current position."""
    return {
        "current": state.current_node_id,
        "visited": state.visited,
        "notes":   state.notes,
    }


def add_note(state: AgentState, driver: Driver, note: str) -> dict:
    """Add an observation note to the agent's memory."""
    state.add_note(note)
    return {"notes": state.notes}


# ── Tool registry (name → callable) ──────────────────────────────────────────

TOOLS = {
    "read_node":       read_node,
    "read_neighbours": read_neighbours,
    "read_incoming":   read_incoming,
    "read_outgoing":   read_outgoing,
    "move_to":         move_to,
    "read_source_code":read_source_code,
    "get_parent_class":get_parent_class,
    "find_path":       find_path,
    "search_node":     search_node,
    "get_call_chain":  get_call_chain,
    "get_callers":     get_callers,
    "history":         history,
    "add_note":        add_note,
}
