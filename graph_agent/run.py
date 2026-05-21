"""
Interactive demo — GraphAgent navigating Spring PetClinic.

Usage:
    python -m graph_agent.run
    python -m graph_agent.run --node owner_ownercontroller_ownercontroller
    python -m graph_agent.run --node <id> --question "What does this class do?"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# ── Allow running from C:\Projet directly ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_agent.agent import GraphAgent
from graph_agent.tools import get_driver, _run


def list_entry_points(driver) -> list[dict]:
    """Return Classes and Files as suggested starting nodes."""
    return _run(driver, """
        MATCH (n)
        WHERE 'Class' IN labels(n) OR 'File' IN labels(n)
        RETURN n.id AS id, n.label AS label, labels(n) AS type
        ORDER BY n.label
        LIMIT 30
    """)


def pick_start_node(driver) -> str:
    entries = list_entry_points(driver)
    print("\n📍 Available starting nodes (Classes & Files):\n")
    for i, e in enumerate(entries):
        print(f"  [{i:2d}] {e['label']:40s}  {e['id']}")
    print()
    choice = input("Enter index or a node id directly: ").strip()
    if choice.isdigit():
        return entries[int(choice)]["id"]
    return choice


def resolve_node_id(driver, name: str) -> str:
    """Resolve a short name (e.g. 'OwnerController') to the full Neo4j node id."""
    # If it already exists as-is, use it
    rows = _run(driver, "MATCH (n {id: $id}) RETURN n.id AS id", id=name)
    if rows:
        return rows[0]["id"]
    # Try case-insensitive label match
    rows = _run(driver,
        "MATCH (n) WHERE n.label = $lbl OR n.label =~ ('(?i)' + $lbl) RETURN n.id AS id LIMIT 1",
        lbl=name)
    if rows:
        return rows[0]["id"]
    # Fallback: substring match on id
    rows = _run(driver,
        "MATCH (n) WHERE n.id ENDS WITH $suffix RETURN n.id AS id LIMIT 1",
        suffix=name.lower())
    if rows:
        return rows[0]["id"]
    return name  # give up, return as-is


import re

def auto_find_start_node(driver, question: str) -> str | None:
    """
    Extract keywords from the question and find the most relevant Class node
    in Neo4j — without any LLM call. This skips one wasted LLM round-trip.
    Priority: Class nodes > File nodes, ranked by label length (shorter = more specific).
    """
    # Extract capitalised words and plain words as candidate keywords
    words = re.findall(r'[A-Z][a-z]+|[a-z]{4,}', question)
    # Deduplicate, keep longest first so 'visit' beats 'vis'
    seen, keywords = set(), []
    for w in sorted(words, key=len, reverse=True):
        if w.lower() not in seen:
            seen.add(w.lower())
            keywords.append(w)

    for kw in keywords:
        rows = _run(driver, """
            MATCH (n)
            WHERE 'Class' IN labels(n)
              AND (toLower(n.label) CONTAINS toLower($kw)
                   OR toLower(n.id) CONTAINS toLower($kw))
            RETURN n.id AS id, n.label AS label
            ORDER BY size(n.label)
            LIMIT 1
        """, kw=kw)
        if rows:
            return rows[0]["id"]
    return None


def main():
    parser = argparse.ArgumentParser(description="GraphAgent interactive demo")
    parser.add_argument("--node",     help="Starting node id (omit for free exploration)", default=None)
    parser.add_argument("--question", help="Question to ask",  default=None)
    parser.add_argument("--model",    help="Model name",       default="qwen2.5-coder:7b")
    parser.add_argument("--api-base", help="API base URL (default: Ollama local). "
                                           "Use for OpenAI-compatible remote endpoints.",
                        default=None)
    args = parser.parse_args()

    print("🔗 Connecting to Neo4j…")
    driver = get_driver()

    question = args.question or input("\n💬 Question: ").strip()
    if not question:
        question = "What is the role of this node? Explore its neighbours and summarise."

    start_node = args.node

    # ── Auto-find start node from question keywords (no LLM call) ────────────
    if not start_node:
        driver2 = get_driver()
        start_node = auto_find_start_node(driver2, question)
        driver2.close()
        if start_node:
            print(f"🔍 Auto-detected start node: {start_node}")
        else:
            print("⚠️  Could not auto-detect a start node — agent will search itself.")

    # ── Resolve short name → real Neo4j id if provided manually ──────────────
    elif start_node:
        driver2 = get_driver()
        resolved = resolve_node_id(driver2, start_node)
        driver2.close()
        if resolved != start_node:
            print(f"   (resolved '{start_node}' → '{resolved}')")
            start_node = resolved

    driver.close()

    if start_node:
        print(f"\n🤖 Starting agent on node: {start_node}")
    else:
        print(f"\n🤖 Starting agent in FREE EXPLORATION mode (no fixed starting node)")
    api_base = args.api_base
    endpoint_label = api_base if api_base else "Ollama (localhost)"
    print(f"   Model    : {args.model}")
    print(f"   Endpoint : {endpoint_label}")
    print(f"   Question : {question}\n")
    print("─" * 60)

    agent = GraphAgent(model=args.model, api_base=api_base)

    try:
        result = agent.run(question, start_node=start_node)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        return

    print("\n" + "═" * 60)
    print("✅ ANSWER\n")
    print(result["answer"])
    print("\n📍 Visited nodes:", " → ".join(([start_node] if start_node else []) + result["visited"]))
    if result["notes"]:
        print("\n📝 Notes:")
        for n in result["notes"]:
            print(f"   • {n}")
    print("═" * 60)

    agent.close()


if __name__ == "__main__":
    main()
