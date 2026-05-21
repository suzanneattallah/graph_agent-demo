"""
Génère automatiquement des traces MLflow pour le projet macrozheng/mall.

Usage :
  python -m demo_client.2_generer_traces
  python demo_client\2_generer_traces.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_agent.agent import GraphAgent

try:
    from .config import AGENT_EXPERIMENT, AGENT_MODEL, MLFLOW_URL, REMOTE_API_BASE
except ImportError:
    from demo_client.config import AGENT_EXPERIMENT, AGENT_MODEL, MLFLOW_URL, REMOTE_API_BASE

QUESTIONS = [
    (
        "What are the main responsibilities of OmsOrderController and which HTTP endpoints does it expose? List all handler methods.",
        "shallow",
    ),
    (
        "What fields and validation constraints does the OmsOrder entity contain? Does it extend any base class?",
        "shallow",
    ),
    (
        "Trace the complete data flow when a customer submits an order through the portal. Follow the execution path from the HTTP entry point through all business logic layers to final data persistence.",
        "multi",
    ),
    (
        "What happens when a product's inventory is reduced during order placement? Trace every class and method involved in the stock management flow from controller to database.",
        "deep",
    ),
    (
        "How does this application handle authentication and security? Identify all security-related classes, how they intercept requests, and how user permissions are verified.",
        "cross",
    ),
    (
        "Which classes implement the MyBatis mapper pattern? List all mapper interfaces, the entities they manage, and how they are injected into the service layer.",
        "cross",
    ),
    (
        "What is the complete flow for product search? Trace from the HTTP request through any search service to the data retrieval mechanism.",
        "cross",
    ),
]


def run_all(model: str = AGENT_MODEL, api_base: str | None = REMOTE_API_BASE) -> list[dict[str, object]]:
    print(f"Génération de {len(QUESTIONS)} traces MLflow pour mall...")
    print(f"Modèle     : {model}")
    print(f"API base   : {api_base or 'Ollama local'}")
    print(f"Experiment : {AGENT_EXPERIMENT}\n")

    agent = GraphAgent(
        model=model,
        api_base=api_base,
        mlflow_uri=MLFLOW_URL,
        experiment=AGENT_EXPERIMENT,
    )
    results: list[dict[str, object]] = []

    try:
        for idx, (question, qtype) in enumerate(QUESTIONS, start=1):
            print(f"{'=' * 80}")
            print(f"[{idx}/{len(QUESTIONS)}] Type: {qtype.upper()}")
            print(f"Question : {question}")
            print(f"{'=' * 80}")

            started = time.time()
            try:
                result = agent.run(question, max_iters=8)
                elapsed = time.time() - started
                visited = len(result.get("visited", []))
                print(f"\n  ✓ Réponse ({elapsed:.1f}s, {visited} noeud(s) visité(s))")
                print(f"  {result['answer'][:500]}...")
                results.append(
                    {
                        "qtype": qtype,
                        "ok": True,
                        "elapsed": elapsed,
                        "n_visited": visited,
                        "question": question,
                    }
                )
            except Exception as exc:
                print(f"  ✗ ERREUR : {type(exc).__name__}: {exc}")
                results.append({"qtype": qtype, "ok": False, "error": str(exc), "question": question})

            print()
            time.sleep(2)
    finally:
        agent.close()

    print(f"\n{'=' * 80}")
    print("RÉSUMÉ")
    print(f"{'=' * 80}")
    successful = [result for result in results if result.get("ok")]
    print(f"Traces générées : {len(successful)}/{len(QUESTIONS)}")
    if successful:
        avg_nodes = sum(float(result["n_visited"]) for result in successful) / len(successful)
        avg_time = sum(float(result["elapsed"]) for result in successful) / len(successful)
        print(f"Noeuds visités (moy.) : {avg_nodes:.1f}")
        print(f"Durée (moy.)          : {avg_time:.1f}s")
    print("\nLance ensuite : python -m demo_client.3_juges_iteratif")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Génère des traces MLflow pour le projet mall")
    parser.add_argument("--model", default=AGENT_MODEL, help="Nom du modèle agent")
    parser.add_argument("--api-base", default=REMOTE_API_BASE, help="Endpoint OpenAI-compatible")
    return parser


def main() -> list[dict[str, object]]:
    args = _build_parser().parse_args()
    return run_all(model=args.model, api_base=args.api_base)


if __name__ == "__main__":
    main()
