"""
Optimisation itérative du prompt de l'agent GraphAgent (GraphExploreSignature).

Les juges (déjà optimisés dans le Prompt Registry) évaluent les réponses de l'agent.
Ollama raffine le prompt de l'agent jusqu'à ce que tous les juges disent SATISFAISANT.
Le prompt optimisé est versionné dans le Prompt Registry MLflow.

Workflow :
1. Charge les Q+A depuis history_iteration_N.json (produit par juges_iteratif.py)
2. Charge les 4 prompts de juges depuis le Prompt Registry (lecture seule)
3. Boucle :
   a. Les 4 juges évaluent chaque réponse de l'agent avec le prompt courant
   b. Si tous SATISFAISANT → convergence
   c. Sinon → Ollama améliore le prompt de l'agent (GraphExploreSignature docstring)
   d. Nouvelle version enregistrée dans le Prompt Registry
   e. Agent re-testé sur une question de validation (optionnel, --retest)
   f. Itération suivante
4. Convergence → applique le nouveau prompt dans agent.py

Usage :
  python -m graph_agent.juges.optimize_agent
  python -m graph_agent.juges.optimize_agent --retest      # re-run l'agent à chaque iter
  python -m graph_agent.juges.optimize_agent --max-iter 5
"""

import json
import re
import sys
import time as _time
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URL              = "http://localhost:5000"
JUDGE_PREFIX            = "graph-agent"
AGENT_PROMPT_NAME       = "graph-agent-SystemPrompt"
AGENT_EXPERIMENT_NAME   = "Agent_Optimization_GraphAgent"
ARTIFACTS_DIR           = Path(__file__).parent / "artifact"

CONVERGENCE_THRESHOLD   = 1.0   # 100% SATISFAISANT requis
MAX_ITERATIONS          = 10

# Modèle pour les juges ET pour l'agent re-test (même Ollama local)
JUDGE_MODEL    = "qwen2.5-coder:7b"
JUDGE_BASE_URL = "http://localhost:11434/v1"
JUDGE_API_KEY  = "ollama"

REFINER_MODEL  = "qwen2.5-coder:7b"   # même modèle par défaut

JUDGES = ["JugeExploration", "JugePrecisionTechnique", "JugeRaisonnement", "JugeAmeliorations"]
VERDICTS = ["SATISFAISANT", "À AMÉLIORER", "INSUFFISANT"]

# ── Prompt initial de l'agent (copie de GraphExploreSignature.__doc__) ────────
# C'est CE texte qui sera optimisé itérativement.
INITIAL_AGENT_PROMPT = """\
You are an expert code analyst navigating a Neo4j knowledge graph of a Java Spring codebase.
You MUST explore the graph by calling tools — do NOT answer from prior knowledge or training.

MANDATORY RULES (violating any rule = invalid answer):
1. Your VERY FIRST action MUST be tool_read_node() to inspect the starting node.
2. After reading any node, ALWAYS call tool_read_neighbours() or tool_read_outgoing() to find
   what connects to it.
3. You MUST call tool_move_to() at least 5 times to visit at least 5 DIFFERENT nodes.
4. Never answer based on what you know about Spring PetClinic — only use tool results.
5. If you need to find a node by name, use tool_search_node(query) first.
6. Follow CALLS and METHOD edges to trace execution flows across layers.
7. Follow EXTENDS and IMPLEMENTS edges to trace class hierarchies.
8. Use tool_read_source_code() on at least one node to verify actual implementation.
9. Only write your final answer after visiting at least 5 distinct nodes.
10. Cite specific node IDs and method names discovered through tools in your answer.
"""

mlflow.set_tracking_uri(MLFLOW_URL)

def _make_judge_client(timeout: float = 180.0) -> OpenAI:
    return OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=timeout)

def _make_refiner_client(timeout: float = 240.0) -> OpenAI:
    return OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=timeout)


# ── LLM helper ────────────────────────────────────────────────────────────────
def _llm_call(client_factory, model: str, messages: list,
              temperature: float = 0.0, max_retries: int = 3) -> str:
    last_error = None
    for attempt in range(1, max_retries + 1):
        content = ""
        try:
            client = client_factory()
            for chunk in client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, stream=True,
            ):
                if chunk.choices:
                    d = chunk.choices[0].delta
                    if d and d.content:
                        content += d.content
            if content.strip():
                return content.strip()
            last_error = "réponse vide"
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {str(e)[:120]}")
        if attempt < max_retries:
            _time.sleep(5)
    raise RuntimeError(f"LLM échoué : {last_error}")


# ── Extraction verdict ────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().upper()


def extract_verdict(text: str) -> str:
    t = _normalize(text)
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*SATISFAISANT", t):
        return "SATISFAISANT"
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", t):
        m = re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", t)
        return "INSUFFISANT" if "INSUFFISANT" in (m.group(1) if m else "") else "À AMÉLIORER"
    if "SATISFAISANT" in t:
        return "SATISFAISANT"
    if "A AMELIORER" in t or "INSUFFISANT" in t:
        return "INSUFFISANT" if "INSUFFISANT" in t else "À AMÉLIORER"
    return "INCONNU"


def extract_recommendation(text: str) -> str:
    m = re.search(r"RECOMMANDATION\s*:?\s*(.*?)(?:\n{2,}|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()[:600]
    return text[:400]


# ── Dataset : chargement depuis history_iteration_N.json ──────────────────────
def load_history() -> tuple[list[dict], list[dict]]:
    """Charge la dernière histoire depuis artifact/. Retourne (items, feedbacks_non_satisfaisant)."""
    # Prend le dernier fichier history_iteration_*.json
    history_files = sorted(ARTIFACTS_DIR.glob("history_iteration_*.json"))
    if not history_files:
        raise FileNotFoundError(
            f"Aucun history_iteration_*.json dans {ARTIFACTS_DIR}. "
            "Lance d'abord : python -m graph_agent.juges.juges_iteratif"
        )
    history_file = history_files[-1]
    print(f"  Chargement de {history_file.name}")
    history = json.loads(history_file.read_text(encoding="utf-8"))

    items, feedbacks = [], []
    total = satisfaisant = 0

    for entry in history:
        q = entry.get("question", "").strip()
        a = entry.get("answer", "").strip()
        if not q or not a:
            continue
        items.append({"question": q, "answer": a, "run_id": entry.get("run_id", "")})

        for judge_name, jdata in entry.get("judges", {}).items():
            verdict_text = jdata.get("verdict", "")
            v_label = extract_verdict(verdict_text)
            reco    = extract_recommendation(verdict_text)
            total += 1
            if v_label == "SATISFAISANT":
                satisfaisant += 1
            else:
                feedbacks.append({
                    "judge":          judge_name,
                    "question":       q,
                    "answer":         a,
                    "verdict":        v_label,
                    "recommendation": reco,
                })

    avg = satisfaisant / total if total else 0.0
    print(f"  {len(items)} Q+A | {satisfaisant}/{total} SATISFAISANT ({avg:.0%})")
    print(f"  {len(feedbacks)} recommandation(s) à traiter")
    return items, feedbacks


# ── Prompt Registry : agent ────────────────────────────────────────────────────
def ensure_agent_prompt_registered() -> int:
    mlf_client = MlflowClient()
    try:
        versions = mlf_client.search_prompt_versions(name=AGENT_PROMPT_NAME)
        if versions:
            latest = max(int(v.version) for v in versions)
            print(f"  ✓ '{AGENT_PROMPT_NAME}' dans le Registry (v{latest})")
            return latest
    except Exception:
        pass
    print(f"  Enregistrement du prompt initial...")
    pv = mlflow.genai.register_prompt(
        name=AGENT_PROMPT_NAME,
        template=INITIAL_AGENT_PROMPT,
        commit_message="Prompt initial GraphExploreSignature (règles 1-10)",
        tags={"source": "GraphExploreSignature.__doc__", "iteration": "0", "status": "initial"},
    )
    print(f"  ✓ '{AGENT_PROMPT_NAME}' v{pv.version} enregistré")
    return int(pv.version)


def load_agent_prompt() -> tuple[str, int]:
    mlf_client = MlflowClient()
    versions = mlf_client.search_prompt_versions(name=AGENT_PROMPT_NAME)
    latest_v = max(int(v.version) for v in versions)
    prompt_obj = mlflow.genai.load_prompt(f"prompts:/{AGENT_PROMPT_NAME}/{latest_v}")
    return prompt_obj.template, latest_v


def save_agent_prompt(template: str, commit_msg: str, iteration: int) -> int:
    pv = mlflow.genai.register_prompt(
        name=AGENT_PROMPT_NAME,
        template=template,
        commit_message=commit_msg,
        tags={"iteration": str(iteration), "status": "optimized"},
    )
    return int(pv.version)


# ── Juges : lecture depuis Registry ───────────────────────────────────────────
def load_judge_prompts() -> dict[str, str]:
    try:
        from .register_juges import FALLBACK_PROMPTS
    except ImportError:
        from register_juges import FALLBACK_PROMPTS

    mlf_client = MlflowClient()
    prompts = {}
    for j in JUDGES:
        registry_name = f"{JUDGE_PREFIX}-{j}"
        try:
            versions = mlf_client.search_prompt_versions(name=registry_name)
            if versions:
                latest_v = max(int(v.version) for v in versions)
                prompt_obj = mlflow.genai.load_prompt(f"prompts:/{registry_name}/{latest_v}")
                prompts[j] = prompt_obj.template
                print(f"  ✓ {j} v{latest_v}")
                continue
        except Exception as e:
            print(f"  [warn] {j} : {e}")
        prompts[j] = FALLBACK_PROMPTS.get(j, "")
    return prompts


# ── Évaluation d'une réponse par tous les juges ───────────────────────────────
def evaluate_item(judge_prompts: dict[str, str], item: dict,
                  trace_info: str = "") -> dict[str, dict]:
    """Retourne {judge_name: {verdict_text, verdict_label, recommendation}}."""
    results = {}
    for judge_name, instructions in judge_prompts.items():
        filled = (
            instructions
            .replace("{{ inputs }}", item["question"])
            .replace("{{ outputs }}", item["answer"])
            .replace("{{ trace }}", trace_info)
        )
        try:
            verdict_text = _llm_call(
                judge_client, JUDGE_MODEL,
                [{"role": "user", "content": filled}],
            )
        except Exception as e:
            verdict_text = f"[ERREUR] {e}"
        results[judge_name] = {
            "verdict_text":   verdict_text,
            "verdict_label":  extract_verdict(verdict_text),
            "recommendation": extract_recommendation(verdict_text),
        }
    return results


# ── Raffinement du prompt de l'agent ─────────────────────────────────────────
def refine_agent_prompt(current_prompt: str, feedbacks: list[dict]) -> str:
    """Réécrit GraphExploreSignature.__doc__ à partir des recommandations des juges."""
    feedback_text = "\n\n".join([
        f"--- Recommandation du {f['judge']} (verdict {f['verdict']}) ---\n"
        f"Question : {f['question'][:200]}\n"
        f"Problème identifié : {f['recommendation']}"
        for f in feedbacks[:8]  # limite pour ne pas dépasser le contexte
    ])

    system_msg = (
        "Tu es un expert en prompt engineering pour des agents LLM de navigation de graphes. "
        "Tu dois améliorer le prompt système d'un agent DSPy ReAct qui explore un graphe Neo4j "
        "de code source Java (Spring PetClinic). "
        "CONTRAINTES ABSOLUES À PRÉSERVER : "
        "(1) Le prompt doit être en ANGLAIS (c'est le prompt système de l'agent). "
        "(2) Il doit maintenir toutes les règles numérotées (1-10 minimum). "
        "(3) La règle 'move_to() at least 5 times' est obligatoire. "
        "(4) La règle 'do NOT answer from prior knowledge' est obligatoire. "
        "(5) Retourne UNIQUEMENT le nouveau prompt, sans bloc de code, sans explication. "
        "Le but est d'améliorer la profondeur d'exploration et la précision des réponses."
    )
    user_msg = (
        f"Prompt actuel de l'agent :\n\n{current_prompt}\n\n"
        f"Recommandations des juges à intégrer :\n\n{feedback_text}\n\n"
        "Réécris le prompt pour corriger les problèmes identifiés."
    )

    try:
        return _llm_call(
            refiner_client, REFINER_MODEL,
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
        )
    except Exception as e:
        print(f"  [raffinement] Échec : {e} → prompt inchangé")
        return current_prompt


# ── Re-test de l'agent avec le nouveau prompt (optionnel) ─────────────────────
def retest_agent(new_prompt: str, test_question: str) -> dict | None:
    """Lance l'agent avec le nouveau prompt sur une question de validation."""
    try:
        # Ajoute le chemin parent pour importer graph_agent
        root = Path(__file__).parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # Patch dynamique du docstring de GraphExploreSignature
        from graph_agent.agent import GraphExploreSignature, GraphAgent
        original_doc = GraphExploreSignature.__doc__
        GraphExploreSignature.__doc__ = new_prompt

        agent = GraphAgent()
        result = agent.run(test_question)
        agent.close()

        GraphExploreSignature.__doc__ = original_doc  # restaure
        return result
    except Exception as e:
        print(f"  [retest] Erreur : {type(e).__name__}: {e}")
        return None


# ── Application du prompt optimisé dans agent.py ──────────────────────────────
def apply_prompt_to_agent(new_prompt: str) -> bool:
    """Remplace GraphExploreSignature.__doc__ dans agent.py."""
    agent_file = Path(__file__).parents[2] / "graph_agent" / "agent.py"
    if not agent_file.exists():
        print(f"  [apply] {agent_file} introuvable")
        return False

    content = agent_file.read_text(encoding="utf-8")

    # Localise le début et la fin du docstring de GraphExploreSignature
    # Structure attendue :
    #   class GraphExploreSignature(dspy.Signature):
    #       """
    #       ...
    #       """
    start_marker = 'class GraphExploreSignature(dspy.Signature):\n    """'
    end_marker   = '    """\n    question'

    start_idx = content.find(start_marker)
    end_idx   = content.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        print("  [apply] Impossible de localiser GraphExploreSignature.__doc__ dans agent.py")
        return False

    # Indente chaque ligne du nouveau prompt avec 4 espaces
    indented = "\n".join(
        ("    " + line if line.strip() else "")
        for line in new_prompt.strip().splitlines()
    )

    before    = content[: start_idx + len(start_marker)]
    after     = content[end_idx:]
    new_content = before + "\n" + indented + "\n" + after

    agent_file.write_text(new_content, encoding="utf-8")
    print(f"  ✓ GraphExploreSignature.__doc__ mis à jour dans agent.py")
    return True


# ── Boucle principale ─────────────────────────────────────────────────────────
def optimize(do_retest: bool = False, max_iterations: int = MAX_ITERATIONS):
    print("═" * 70)
    print("  OPTIMISATION DU PROMPT GRAPHAGENT")
    print("═" * 70)

    # Chargement
    print("\n1. Chargement du dataset...")
    items, initial_feedbacks = load_history()
    if not items:
        print("  Aucun item chargé. Lance d'abord juges_iteratif.py.")
        return

    print("\n2. Chargement des juges...")
    judge_prompts = load_judge_prompts()
    if not judge_prompts:
        print("  Aucun juge disponible. Lance d'abord register_juges.py.")
        return

    print("\n3. Chargement du prompt de l'agent...")
    ensure_agent_prompt_registered()
    agent_prompt, current_version = load_agent_prompt()
    print(f"  Prompt v{current_version} chargé ({len(agent_prompt)} chars)")

    mlflow.set_experiment(AGENT_EXPERIMENT_NAME)

    # Question de validation pour le re-test
    validation_question = (
        "Trace the complete call chain from VisitController to the database "
        "when saving a new visit. Identify all classes, methods, and layers involved."
    )

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"  ITÉRATION {iteration}/{max_iterations}")
        print(f"{'='*70}")

        with mlflow.start_run(run_name=f"optimize_iter_{iteration}") as run:
            mlflow.log_params({
                "iteration":      iteration,
                "n_items":        len(items),
                "agent_prompt_v": current_version,
                "judge_model":    JUDGE_MODEL,
            })
            mlflow.log_text(agent_prompt, f"agent_prompt_iter_{iteration}.txt")

            # Évaluation de chaque item par tous les juges
            all_verdicts = []
            feedbacks_to_fix = []

            for i, item in enumerate(items):
                print(f"\n  Évaluation {i+1}/{len(items)}: {item['question'][:80]}...")
                verdicts = evaluate_item(judge_prompts, item)

                for judge_name, vdata in verdicts.items():
                    v = vdata["verdict_label"]
                    all_verdicts.append(v)
                    print(f"    {judge_name}: {v}")
                    if v != "SATISFAISANT":
                        feedbacks_to_fix.append({
                            "judge":          judge_name,
                            "question":       item["question"],
                            "answer":         item["answer"],
                            "verdict":        v,
                            "recommendation": vdata["recommendation"],
                        })

            # Score de convergence
            satisfaisant_count = all_verdicts.count("SATISFAISANT")
            score = satisfaisant_count / len(all_verdicts) if all_verdicts else 0.0
            print(f"\n  Score SATISFAISANT : {satisfaisant_count}/{len(all_verdicts)} ({score:.0%})")

            mlflow.log_metric("satisfaisant_score", score, step=iteration)
            mlflow.log_metric("n_feedbacks_to_fix", len(feedbacks_to_fix), step=iteration)

            if score >= CONVERGENCE_THRESHOLD:
                print(f"\n✅ CONVERGENCE à l'itération {iteration} — score : {score:.0%}")
                mlflow.log_param("converged_at", iteration)
                break

            if not feedbacks_to_fix:
                print("  Aucun feedback à corriger malgré score < 100%.")
                break

            # Raffinement du prompt
            print(f"\n  Raffinement du prompt ({len(feedbacks_to_fix)} problème(s))...")
            new_prompt = refine_agent_prompt(agent_prompt, feedbacks_to_fix)

            # Sauvegarde dans le Registry
            commit_msg = f"Iter {iteration}: {score:.0%} SATISFAISANT → raffinement"
            new_version = save_agent_prompt(new_prompt, commit_msg, iteration)
            print(f"  ✓ Nouvelle version : {AGENT_PROMPT_NAME} v{new_version}")
            mlflow.log_metric("new_prompt_version", new_version, step=iteration)

            agent_prompt     = new_prompt
            current_version  = new_version

            # Re-test optionnel
            if do_retest:
                print(f"\n  Re-test de l'agent sur la question de validation...")
                retest_result = retest_agent(new_prompt, validation_question)
                if retest_result:
                    n_visited = len(retest_result.get("visited", []))
                    print(f"  ✓ Re-test OK — {n_visited} noeuds visités")
                    mlflow.log_metric("retest_n_visited", n_visited, step=iteration)
                else:
                    print("  ✗ Re-test échoué (agent ou Neo4j non disponible)")

    else:
        print(f"\n⚠️  Max itérations ({MAX_ITERATIONS}) atteint sans convergence complète.")

    # Application finale dans agent.py
    print(f"\n{'─'*70}")
    print("  APPLICATION DU PROMPT OPTIMISÉ DANS agent.py")
    print(f"{'─'*70}")
    if apply_prompt_to_agent(agent_prompt):
        print(f"  Prompt v{current_version} appliqué. Relance l'agent pour valider.")
    else:
        print(f"  Copie manuelle : prompts/{AGENT_PROMPT_NAME}/{current_version}")

    print("\n✅ Optimisation terminée.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimise le prompt GraphExploreSignature via les juges")
    parser.add_argument("--retest",   action="store_true", help="Re-run l'agent à chaque itération (lent)")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Nb max d'itérations")
    args = parser.parse_args()
    optimize(do_retest=args.retest, max_iterations=args.max_iter)
