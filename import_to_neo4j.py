"""Import AST nodes+edges JSON into Neo4j Aura."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from neo4j import GraphDatabase

# ── Neo4j connection ──────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "agent_recherche_neo4j"
NEO4J_DATABASE = "neo4j"

# ── Input JSON ────────────────────────────────────────────────────────────────
AST_JSON = Path(r"C:\Projet\mall-ast.json")


def infer_label(node: dict) -> str:
    """Derive a Neo4j node label from the AST node fields."""
    label: str = node.get("label", "")
    src: str   = node.get("source_file", "")

    if label.endswith(".java"):
        return "File"
    if label.startswith("."):
        return "Method"
    if src:                      # defined inside the scanned codebase
        return "Class"
    return "External"            # imported / referenced symbol


def run_import(data: dict, driver) -> None:
    nodes = data["nodes"]
    edges = data["edges"]

    print(f"Importing {len(nodes)} nodes and {len(edges)} edges …")

    with driver.session(database=NEO4J_DATABASE) as session:
        # ── Clear existing data ───────────────────────────────────────────────
        session.run("MATCH (n) DETACH DELETE n")
        print("  ✓ cleared existing graph")

        # ── Create constraints / indexes ──────────────────────────────────────
        for lbl in ("File", "Class", "Method", "External"):
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.id IS UNIQUE"
            )

        # ── Batch-insert nodes (grouped by label for typed MERGE) ─────────────
        BATCH = 500
        from collections import defaultdict
        by_lbl: dict[str, list[dict]] = defaultdict(list)
        for n in nodes:
            by_lbl[infer_label(n)].append(n)

        total_nodes = 0
        for lbl, lbl_nodes in by_lbl.items():
            for i in range(0, len(lbl_nodes), BATCH):
                batch = lbl_nodes[i : i + BATCH]
                params = [
                    {
                        "id":              n["id"],
                        "label":           n.get("label", ""),
                        "source_file":     n.get("source_file", ""),
                        "source_location": n.get("source_location", ""),
                        "file_type":       n.get("file_type", ""),
                    }
                    for n in batch
                ]
                session.run(
                    f"""
                    UNWIND $nodes AS props
                    MERGE (n:{lbl} {{id: props.id}})
                    SET n.label           = props.label,
                        n.source_file     = props.source_file,
                        n.source_location = props.source_location,
                        n.file_type       = props.file_type
                    """,
                    nodes=params,
                )
            total_nodes += len(lbl_nodes)
            print(f"  ✓ {len(lbl_nodes):4d}  {lbl}")
        print(f"  ✓ {total_nodes} nodes total")

        # ── Batch-insert edges ────────────────────────────────────────────────
        # Group by relation type so we can use a typed MERGE in one query each
        by_rel: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            by_rel[e["relation"]].append(e)

        total_edges = 0
        for rel_type, rel_edges in by_rel.items():
            neo4j_rel = rel_type.upper()
            for i in range(0, len(rel_edges), BATCH):
                batch = rel_edges[i : i + BATCH]
                params = [
                    {
                        "src":      e["source"],
                        "tgt":      e["target"],
                        "context":  e.get("context", ""),
                        "weight":   e.get("weight", 1.0),
                        "confidence": e.get("confidence", ""),
                        "source_file": e.get("source_file", ""),
                        "source_location": e.get("source_location", ""),
                    }
                    for e in batch
                ]
                session.run(
                    f"""
                    UNWIND $edges AS e
                    MATCH (src {{id: e.src}})
                    MATCH (tgt {{id: e.tgt}})
                    MERGE (src)-[r:{neo4j_rel}]->(tgt)
                    SET r.context  = e.context,
                        r.weight   = e.weight,
                        r.confidence = e.confidence,
                        r.source_file = e.source_file,
                        r.source_location = e.source_location
                    """,
                    edges=params,
                )
            total_edges += len(rel_edges)
            print(f"  ✓ {len(rel_edges):4d}  {neo4j_rel}")

        print(f"  ✓ {total_edges} edges total")

    print("\n✅ Import complete!")


def main() -> None:
    if not AST_JSON.exists():
        print(f"error: {AST_JSON} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(AST_JSON.read_text(encoding="utf-8"))
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"✓ Connected to {NEO4J_URI}")
        run_import(data, driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
