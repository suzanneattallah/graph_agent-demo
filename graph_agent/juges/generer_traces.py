"""
Génère automatiquement des traces MLflow en posant plusieurs questions à l'agent GraphAgent.

Questions calibrées pour tester les 4 juges :
  - "shallow"   : questions sur un seul composant (teste si l'agent explore quand même en profondeur)
  - "multi"     : questions multi-layers (Controller→Service→Repo→Model)
  - "cross"     : questions architecturales transversales
  - "deep"      : tracé complet de bout en bout (question de référence)

Usage : python -m graph_agent.juges.generer_traces
"""

import sys
import time
from pathlib import Path

# Ajoute C:\Projet au path pour importer graph_agent
sys.path.insert(0, str(Path(__file__).parents[2]))

from graph_agent.agent import GraphAgent

# ── Questions calibrées ───────────────────────────────────────────────────────
# Format : (question, type_attendu)
QUESTIONS = [
    # ── SHALLOW : un seul composant demandé, mais l'agent doit quand même creuser ──
    (
        "What are the main responsibilities of the VisitController class "
        "and which HTTP endpoints does it expose? List all handler methods with their mappings.",
        "shallow",
    ),
    (
        "What fields and validation annotations does the Visit entity class contain? "
        "Does it extend or implement any base class?",
        "shallow",
    ),

    # ── MULTI-LAYER : flow complet attendu ─────────────────────────────────────
    (
        "What is the complete data flow when a user creates a new pet owner "
        "through the web interface? Trace the full execution path from the HTTP "
        "layer to the database, identifying all classes and methods involved.",
        "multi",
    ),
    (
        "Trace the complete call chain from VisitController to the database "
        "when saving a new visit. Identify the controller method, any service "
        "layer, the repository interface, and how persistence is ultimately achieved.",
        "deep",
    ),

    # ── CROSS-CUTTING : plusieurs packages / aspects ───────────────────────────
    (
        "How does Spring PetClinic handle form validation? "
        "Identify all validator classes, where they are registered, "
        "and which controller methods invoke them.",
        "cross",
    ),
    (
        "What is the class hierarchy of entity classes in PetClinic? "
        "Identify all classes that extend a base entity, list the shared "
        "fields and methods, and explain the inheritance strategy.",
        "cross",
    ),
    (
        "Which classes implement the Spring Data repository pattern? "
        "List all repository interfaces, the entity types they manage, "
        "and how they are injected into the service or controller layer.",
        "cross",
    ),
]


def run_all(model: str = "qwen2.5-coder:7b"):
    print(f"Génération de {len(QUESTIONS)} traces MLflow...")
    print(f"Modèle : {model}\n")

    agent = GraphAgent(model=model)
    results = []

    for idx, (question, qtype) in enumerate(QUESTIONS, 1):
        print(f"{'='*70}")
        print(f"[{idx}/{len(QUESTIONS)}] Type: {qtype.upper()}")
        print(f"Question : {question[:120]}...")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = agent.run(question)
            elapsed = time.time() - t0
            n_visited = len(result.get("visited", []))

            print(f"\n  ✓ Réponse ({elapsed:.1f}s, {n_visited} noeuds visités)")
            print(f"  {result['answer'][:400]}...")
            results.append({"qtype": qtype, "ok": True, "elapsed": elapsed, "n_visited": n_visited})
        except Exception as e:
            print(f"  ✗ ERREUR : {type(e).__name__}: {e}")
            results.append({"qtype": qtype, "ok": False, "error": str(e)})

        print()
        # Pause courte pour éviter de surcharger Ollama
        time.sleep(2)

    agent.close()

    # Résumé
    print(f"\n{'='*70}")
    print("  RÉSUMÉ")
    print(f"{'='*70}")
    ok = [r for r in results if r.get("ok")]
    print(f"  Traces générées : {len(ok)}/{len(QUESTIONS)}")
    if ok:
        avg_nodes = sum(r["n_visited"] for r in ok) / len(ok)
        avg_time  = sum(r["elapsed"] for r in ok) / len(ok)
        print(f"  Noeuds visités (moy.) : {avg_nodes:.1f}")
        print(f"  Durée (moy.)          : {avg_time:.1f}s")
    print(f"\n  Lance ensuite : python -m graph_agent.juges.juges_iteratif")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génère des traces MLflow pour GraphAgent")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Modèle Ollama")
    args = parser.parse_args()
    run_all(model=args.model)
