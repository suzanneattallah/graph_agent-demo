from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
except ImportError:
    from demo_client.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

NODE_LABELS = ["Class", "Method", "File", "External"]
REL_TYPES = ["METHOD", "CALLS", "IMPORTS", "CONTAINS", "EXTENDS", "IMPLEMENTS"]


def _format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def fmt(row: list[object]) -> str:
        return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join([fmt(headers), divider, *[fmt(row) for row in rows]])


def fetch_graph_stats() -> dict[str, object]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database="neo4j") as session:
            node_counts = {
                row["label"]: row["count"]
                for row in session.run(
                    "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
                )
            }
            edge_counts = {
                row["rel_type"]: row["count"]
                for row in session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count"
                )
            }
            top_classes = [
                {
                    "label": row["label"] or "(no label)",
                    "id": row["id"],
                    "degree": row["degree"],
                }
                for row in session.run(
                    """
                    MATCH (c:Class)
                    RETURN c.label AS label, c.id AS id, COUNT { (c)--() } AS degree
                    ORDER BY degree DESC, label ASC
                    LIMIT 10
                    """
                )
            ]
    finally:
        driver.close()

    return {
        "node_counts": {label: int(node_counts.get(label, 0)) for label in NODE_LABELS},
        "edge_counts": {rel: int(edge_counts.get(rel, 0)) for rel in REL_TYPES},
        "top_classes": top_classes,
    }


def show_graph_stats() -> dict[str, object]:
    stats = fetch_graph_stats()

    print("=" * 88)
    print("Mall Neo4j Graph Statistics")
    print("=" * 88)
    print(f"URI      : {NEO4J_URI}")
    print(f"Database : neo4j\n")

    node_rows = [[label, stats["node_counts"][label]] for label in NODE_LABELS]
    print("Nodes by label")
    print(_format_table(["Label", "Count"], node_rows))
    print()

    edge_rows = [[rel.lower(), stats["edge_counts"][rel]] for rel in REL_TYPES]
    print("Edges by relation type")
    print(_format_table(["Relation", "Count"], edge_rows))
    print()

    top_rows = [
        [index, item["label"], item["id"], item["degree"]]
        for index, item in enumerate(stats["top_classes"], start=1)
    ]
    print("Top 10 most connected classes")
    print(_format_table(["#", "Class", "Node ID", "Degree"], top_rows))
    return stats


def main() -> dict[str, object]:
    return show_graph_stats()


if __name__ == "__main__":
    main()
