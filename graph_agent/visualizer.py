"""
Real-time graph navigation visualizer for GraphAgent.

Run with:
    streamlit run C:/Projet/graph_agent/visualizer.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# ── Config ────────────────────────────────────────────────────────────────────
NAV_STATE_FILE = Path(__file__).parent / "nav_state.json"
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "agent_recherche_neo4j"

# ── Package color palette (vivid, dark-bg friendly) ───────────────────────────
COLOR_CURRENT  = "#14b8a6"   # teal  — agent is here
COLOR_VISITED  = "#7c3aed"   # violet — already visited
COLOR_NODE     = "#94a3b8"   # grey   — all other nodes
COLOR_NAV_EDGE = "#14b8a6"   # teal   — traversed path
COLOR_BG       = "#ffffff"   # white background


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GraphAgent Navigator", layout="wide", page_icon="🤖")
st.title("🤖 GraphAgent — Real-time Navigation")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_nav_state() -> dict:
    if NAV_STATE_FILE.exists():
        try:
            return json.loads(NAV_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"current": None, "visited": [], "notes": [], "moves": []}


def load_subgraph(nav: dict) -> tuple[list, list]:
    """
    Load only the relevant subgraph from Neo4j:
    - current node + visited nodes
    - 1-hop neighbours of the current node
    This keeps the graph small enough to render regardless of codebase size.
    """
    current      = nav.get("current")
    visited_list = nav.get("visited", [])
    focal_ids    = list({current} | set(visited_list) - {None})

    if not focal_ids:
        return [], []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database="neo4j") as s:
            # Focal nodes + their direct neighbours (1 hop)
            raw_nodes = s.run("""
                MATCH (n)
                WHERE n.id IN $ids
                WITH collect(n) AS focal
                UNWIND focal AS f
                OPTIONAL MATCH (f)-[r]-(nb)
                WITH focal, collect(DISTINCT nb) AS neighbours
                WITH focal + neighbours AS all_nodes
                UNWIND all_nodes AS n
                RETURN DISTINCT
                    n.id AS id, n.label AS label,
                    labels(n) AS types, n.source_file AS source_file
            """, ids=focal_ids).data()

            # Only edges between nodes in our subgraph
            subgraph_ids = [n["id"] for n in raw_nodes if n["id"]]
            raw_edges = s.run("""
                MATCH (a)-[r]->(b)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS src, b.id AS tgt, type(r) AS rel
            """, ids=subgraph_ids).data()
    finally:
        driver.close()

    return raw_nodes, raw_edges


TOOL_ICONS = {
    "tool_read_node":         "🔍",
    "tool_read_neighbours":   "🕸️",
    "tool_read_incoming":     "⬅️",
    "tool_read_outgoing":     "➡️",
    "tool_move_to":           "🔀",
    "tool_read_source_code":  "📄",
    "tool_get_parent_class":  "👆",
    "tool_find_path":         "🗺️",
    "tool_search_node":       "🔎",
    "tool_get_call_chain":    "⛓️",
    "tool_get_callers":       "📞",
    "tool_history":           "📋",
    "tool_add_note":          "📝",
}


def build_agraph(raw_nodes, raw_edges, nav: dict):
    current      = nav.get("current")
    visited_list = nav.get("visited", [])
    visited_set  = set(visited_list)
    full_path    = visited_list + ([current] if current else [])
    traversed    = set(zip(full_path, full_path[1:]))

    nodes, edges = [], []

    for n in raw_nodes:
        nid   = n["id"]
        label = n["label"] or nid
        types = n.get("types", [])
        src   = n.get("source_file") or ""

        if nid == current:
            color       = {"background": "#ffffff", "border": COLOR_CURRENT,
                           "highlight": {"background": "#ccfbf1", "border": COLOR_CURRENT}}
            size, shape = 26, "hexagon"
            font        = {"size": 14, "color": "#0f4f4f", "bold": True}

        elif nid in visited_set:
            color       = {"background": COLOR_VISITED, "border": "#5b21b6",
                           "highlight": {"background": "#a78bfa", "border": "#5b21b6"}}
            size, shape = 20, "dot"
            font        = {"size": 12, "color": "#1e1b4b", "bold": True}

        elif not src:
            color       = {"background": "#e2e8f0", "border": "#e2e8f0"}
            size, shape = 4, "dot"
            font        = {"size": 0, "color": "transparent"}

        elif "Method" in types:
            color       = {"background": "#cbd5e1", "border": "#cbd5e1"}
            size, shape = 5, "dot"
            font        = {"size": 0, "color": "transparent"}

        elif "File" in types:
            color       = {"background": "#94a3b8", "border": "#64748b",
                           "highlight": {"background": "#94a3b8", "border": "#475569"}}
            size, shape = 12, "box"
            font        = {"size": 9, "color": "#1e293b"}

        else:
            color       = {"background": COLOR_NODE, "border": "#64748b",
                           "highlight": {"background": "#cbd5e1", "border": "#475569"}}
            size, shape = 16, "dot"
            font        = {"size": 10, "color": "#1e293b"}

        nodes.append(Node(
            id=nid,
            label=label[:30] if (nid in visited_set or nid == current
                                  or "Class" in types or "File" in types) else "",
            size=size,
            color=color,
            shape=shape,
            font=font,
        ))

    for e in raw_edges:
        is_nav = (e["src"], e["tgt"]) in traversed
        edges.append(Edge(
            source=e["src"],
            target=e["tgt"],
            label=e["rel"] if is_nav else "",
            color=COLOR_NAV_EDGE if is_nav else "#cbd5e1",
            width=3.5 if is_nav else 0.5,
        ))

    # ── Virtual navigation arrows between consecutive visited nodes ────────────
    # These show the agent's path even when nodes aren't directly linked in Neo4j
    moves_lookup = {(m["from"], m["to"]): m.get("via", "nav")
                    for m in nav.get("moves", [])}
    subgraph_ids = {n["id"] for n in raw_nodes}
    existing_edges = {(e["src"], e["tgt"]) for e in raw_edges}
    for src, tgt in traversed:
        if src in subgraph_ids and tgt in subgraph_ids \
                and (src, tgt) not in existing_edges:
            via_label = moves_lookup.get((src, tgt), "nav").replace("tool_", "")
            edges.append(Edge(
                source=src,
                target=tgt,
                label=via_label,
                color="#f59e0b",   # amber — virtual nav edge
                width=2.5,
                dashes=True,
            ))

    return nodes, edges


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗺️ Navigation state")
    auto_refresh = st.toggle("Auto-refresh (1s)", value=True)
    st.divider()

    nav         = load_nav_state()
    current     = nav.get("current")
    visited     = nav.get("visited", [])
    notes       = nav.get("notes", [])
    full_path   = visited + ([current] if current else [])

    if current:
        st.markdown("**📍 Current node**")
        st.code(current, language=None)
    else:
        st.info("Waiting for agent…")

    if full_path:
        st.markdown(f"**🔄 Path** — {len(full_path)} nodes")
        for i, node in enumerate(full_path):
            icon  = "🟢" if node == current else "🔵"
            arrow = "  ↓" if i < len(full_path) - 1 else ""
            st.markdown(f"{icon} `{node}`{arrow}")

    if notes:
        st.divider()
        st.markdown("**📝 Notes**")
        for note in notes:
            st.caption(f"• {note}")

    # ── Tool call feed ────────────────────────────────────────────────────────
    tool_calls = nav.get("tool_calls", [])
    if tool_calls:
        st.divider()
        st.markdown(f"**⚙️ Tool calls** — {len(tool_calls)} total")
        for tc in reversed(tool_calls[-30:]):   # show last 30
            tool = tc.get("tool", "?")
            icon = TOOL_ICONS.get(tool, "⚙️")
            node_short = (tc.get("node") or "?").split("_")[-1]
            args = tc.get("args", {})
            # Build a short args summary
            arg_str = ""
            if "node_id" in args and args["node_id"]:
                arg_str = f' `{args["node_id"].split("_")[-1]}`'
            elif "name" in args:
                arg_str = f' `{args["name"]}`'
            elif "relation" in args and args["relation"]:
                arg_str = f' `{args["relation"]}`'
            elif "target_id" in args:
                arg_str = f' `{args["target_id"].split("_")[-1]}`'
            elif "note" in args:
                arg_str = f' _{args["note"][:40]}…_'
            st.markdown(
                f"{icon} **{tool.replace('tool_', '')}**{arg_str}  \n"
                f"<span style='color:#94a3b8;font-size:0.72em'>@ {node_short}</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("**Legend**")
    st.markdown('<span style="color:#14b8a6">⬡</span> **Current** (white hexagon)', unsafe_allow_html=True)
    st.markdown('<span style="color:#7c3aed">●</span> **Visited** (violet)', unsafe_allow_html=True)
    st.markdown('<span style="color:#14b8a6">—</span> Traversed edge (graph)', unsafe_allow_html=True)
    st.markdown('<span style="color:#f59e0b">- -</span> Navigation jump (virtual)', unsafe_allow_html=True)
    st.markdown('<span style="color:#94a3b8">●</span> Neighbour node (grey)', unsafe_allow_html=True)

# ── Graph ─────────────────────────────────────────────────────────────────────
raw_nodes, raw_edges = load_subgraph(nav)
nodes, edges = build_agraph(raw_nodes, raw_edges, nav)

st.markdown(
    f"<span style='color:#475569'>{len(nodes)} nodes · {len(edges)} edges · "
    f"Agent at: <code style='color:#14b8a6'>{current or '—'}</code> · "
    f"Moves: <b>{len(visited)}</b></span>",
    unsafe_allow_html=True,
)

config = Config(
    width="100%",
    height=740,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor=COLOR_CURRENT,
    backgroundColor=COLOR_BG,
    collapsible=False,
)

agraph(nodes=nodes, edges=edges, config=config)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(1)
    st.rerun()
