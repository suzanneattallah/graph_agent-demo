"""
Optimisation itérative du prompt de l'agent GraphAgent pour le projet mall.

Usage :
  python -m demo_client.4_optimize_agent
  python demo_client\4_optimize_agent.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import (
        AGENT_EXPERIMENT,
        AGENT_MODEL,
        AGENT_PROMPT_NAME,
        API_KEY,
        ARTIFACTS_DIR,
        CONVERGENCE_THRESHOLD,
        JUDGES,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        MAX_ITERATIONS,
        MLFLOW_URL,
        REFINER_MODEL,
        REMOTE_API_BASE,
    )
except ImportError:
    from demo_client.config import (
        AGENT_EXPERIMENT,
        AGENT_MODEL,
        AGENT_PROMPT_NAME,
        API_KEY,
        ARTIFACTS_DIR,
        CONVERGENCE_THRESHOLD,
        JUDGES,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        MAX_ITERATIONS,
        MLFLOW_URL,
        REFINER_MODEL,
        REMOTE_API_BASE,
    )

from graph_agent.juges.register_juges import FALLBACK_PROMPTS

AGENT_EXPERIMENT_NAME = "demo-client-agent-optimization"
VERDICTS = ["SATISFAISANT", "À AMÉLIORER", "INSUFFISANT"]
INITIAL_AGENT_PROMPT = """\
You are an expert code analyst navigating a Neo4j knowledge graph of the macrozheng/mall Java Spring Boot e-commerce codebase.
You MUST explore the graph by calling tools — do NOT answer from prior knowledge or training.

MANDATORY RULES (violating any rule = invalid answer):
1. Your VERY FIRST action MUST be tool_read_node() to inspect the starting node.
2. After reading any node, ALWAYS call tool_read_neighbours() or tool_read_outgoing() to find what connects to it.
3. You MUST call tool_move_to() at least 5 times to visit at least 5 DIFFERENT nodes.
4. Never answer based on what you know about the mall e-commerce project — only use tool results.
5. If you need to find a node by name, use tool_search_node(query) first.
6. Follow CALLS and METHOD edges to trace execution flows across controllers, services, mappers, and persistence.
7. Follow EXTENDS and IMPLEMENTS edges to trace class hierarchies and framework integrations.
8. Use tool_read_source_code() on at least one node to verify the actual implementation.
9. Only write your final answer after visiting at least 5 distinct nodes.
10. Cite specific node IDs and method names discovered through tools in your answer.
"""

mlflow.set_tracking_uri(MLFLOW_URL)

def _make_client(timeout: float = 180.0) -> OpenAI:
    return OpenAI(base_url=REMOTE_API_BASE, api_key=API_KEY, timeout=timeout)


def _llm_call(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float = 180.0,
) -> str:
    last_error: object = None
    for attempt in range(1, max_retries + 1):
        content = ""
        try:
            client = _make_client(timeout=timeout)
            for chunk in client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content
            if content.strip():
                return content.strip()
            last_error = "réponse vide"
        except Exception as exc:
            last_error = exc
            print(f"    [retry {attempt}/{max_retries}] {type(exc).__name__}: {str(exc)[:120]}")
        if attempt < max_retries:
            _time.sleep(5)
    raise RuntimeError(f"LLM échoué : {last_error}")


def _normalize(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode().upper()


def extract_verdict(text: str) -> str:
    normalized = _normalize(text)
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*SATISFAISANT", normalized):
        return "SATISFAISANT"
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", normalized):
        match = re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", normalized)
        return "INSUFFISANT" if "INSUFFISANT" in (match.group(1) if match else "") else "À AMÉLIORER"
    if "SATISFAISANT" in normalized:
        return "SATISFAISANT"
    if "A AMELIORER" in normalized or "INSUFFISANT" in normalized:
        return "INSUFFISANT" if "INSUFFISANT" in normalized else "À AMÉLIORER"
    return "INCONNU"


def extract_recommendation(text: str) -> str:
    match = re.search(r"RECOMMANDATION\s*:?\s*(.*?)(?:\n{2,}|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:600]
    return text[:400]


def _history_sort_key(path: Path) -> int:
    match = re.search(r"history_iteration_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def load_history() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    history_files = sorted(ARTIFACTS_DIR.glob("history_iteration_*.json"), key=_history_sort_key)
    if not history_files:
        raise FileNotFoundError(
            f"Aucun history_iteration_*.json dans {ARTIFACTS_DIR}. Lance d'abord : python -m demo_client.3_juges_iteratif"
        )

    history_file = history_files[-1]
    print(f"  Chargement de {history_file.name}")
    history = json.loads(history_file.read_text(encoding="utf-8"))

    items: list[dict[str, str]] = []
    feedbacks: list[dict[str, str]] = []
    total = 0
    satisfaisant = 0

    for entry in history:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if not question or not answer:
            continue
        items.append({"question": question, "answer": answer, "run_id": entry.get("run_id", "")})

        for judge_name, judge_data in entry.get("judges", {}).items():
            verdict_text = judge_data.get("verdict", "")
            verdict_label = extract_verdict(verdict_text)
            recommendation = extract_recommendation(verdict_text)
            total += 1
            if verdict_label == "SATISFAISANT":
                satisfaisant += 1
            else:
                feedbacks.append(
                    {
                        "judge": judge_name,
                        "question": question,
                        "answer": answer,
                        "verdict": verdict_label,
                        "recommendation": recommendation,
                    }
                )

    average = satisfaisant / total if total else 0.0
    print(f"  {len(items)} Q+A | {satisfaisant}/{total} SATISFAISANT ({average:.0%})")
    print(f"  {len(feedbacks)} recommandation(s) à traiter")
    return items, feedbacks


def ensure_agent_prompt_registered() -> int:
    mlf_client = MlflowClient()
    try:
        versions = mlf_client.search_prompt_versions(name=AGENT_PROMPT_NAME)
        if versions:
            latest = max(int(version.version) for version in versions)
            print(f"  ✓ '{AGENT_PROMPT_NAME}' dans le Registry (v{latest})")
            return latest
    except Exception:
        pass

    print("  Enregistrement du prompt initial...")
    prompt_version = mlflow.genai.register_prompt(
        name=AGENT_PROMPT_NAME,
        template=INITIAL_AGENT_PROMPT,
        commit_message="Prompt initial GraphExploreSignature pour mall",
        tags={"source": "demo_client", "iteration": "0", "status": "initial"},
    )
    print(f"  ✓ '{AGENT_PROMPT_NAME}' v{prompt_version.version} enregistré")
    return int(prompt_version.version)


def load_agent_prompt() -> tuple[str, int]:
    mlf_client = MlflowClient()
    versions = mlf_client.search_prompt_versions(name=AGENT_PROMPT_NAME)
    latest_version = max(int(version.version) for version in versions)
    prompt_obj = mlflow.genai.load_prompt(f"prompts:/{AGENT_PROMPT_NAME}/{latest_version}")
    return prompt_obj.template, latest_version


def save_agent_prompt(template: str, commit_message: str, iteration: int) -> int:
    prompt_version = mlflow.genai.register_prompt(
        name=AGENT_PROMPT_NAME,
        template=template,
        commit_message=commit_message,
        tags={"iteration": str(iteration), "status": "optimized"},
    )
    return int(prompt_version.version)


def load_judge_prompts() -> dict[str, str]:
    mlf_client = MlflowClient()
    prompts: dict[str, str] = {}
    for judge_name in JUDGES:
        registry_name = f"{JUDGE_PREFIX}-{judge_name}"
        try:
            versions = mlf_client.search_prompt_versions(name=registry_name)
            if versions:
                latest_version = max(int(version.version) for version in versions)
                prompt_obj = mlflow.genai.load_prompt(f"prompts:/{registry_name}/{latest_version}")
                prompts[judge_name] = prompt_obj.template
                print(f"  ✓ {judge_name} v{latest_version}")
                continue
        except Exception as exc:
            print(f"  [warn] {judge_name} : {exc}")
        prompts[judge_name] = FALLBACK_PROMPTS.get(judge_name, "")
    return prompts


def evaluate_item(judge_prompts: dict[str, str], item: dict[str, str], trace_info: str = "") -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for judge_name, instructions in judge_prompts.items():
        filled = instructions.replace("{{ inputs }}", item["question"]).replace("{{ outputs }}", item["answer"]).replace("{{ trace }}", trace_info)
        try:
            verdict_text = _llm_call(JUDGE_MODEL, [{"role": "user", "content": filled}])
        except Exception as exc:
            verdict_text = f"[ERREUR] {exc}"
        results[judge_name] = {
            "verdict_text": verdict_text,
            "verdict_label": extract_verdict(verdict_text),
            "recommendation": extract_recommendation(verdict_text),
        }
    return results


def refine_agent_prompt(current_prompt: str, feedbacks: list[dict[str, str]]) -> str:
    feedback_text = "\n\n".join(
        [
            f"--- Recommandation du {feedback['judge']} (verdict {feedback['verdict']}) ---\n"
            f"Question : {feedback['question'][:200]}\n"
            f"Problème identifié : {feedback['recommendation']}"
            for feedback in feedbacks[:8]
        ]
    )

    system_msg = (
        "Tu es un expert en prompt engineering pour des agents LLM de navigation de graphes. "
        "Tu dois améliorer le prompt système d'un agent DSPy ReAct qui explore un graphe Neo4j "
        "de code source Java du projet macrozheng/mall (plateforme e-commerce Spring Boot). "
        "CONTRAINTES ABSOLUES À PRÉSERVER : "
        "(1) Le prompt doit être en ANGLAIS. "
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
            REFINER_MODEL,
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
            timeout=240.0,
        )
    except Exception as exc:
        print(f"  [raffinement] Échec : {exc} → prompt inchangé")
        return current_prompt


def retest_agent(new_prompt: str, test_question: str) -> dict[str, object] | None:
    agent = None
    try:
        from graph_agent.agent import GraphAgent, GraphExploreSignature

        original_doc = GraphExploreSignature.__doc__
        GraphExploreSignature.__doc__ = new_prompt
        try:
            agent = GraphAgent(
                model=AGENT_MODEL,
                api_base=REMOTE_API_BASE,
                mlflow_uri=MLFLOW_URL,
                experiment=AGENT_EXPERIMENT,
            )
            return agent.run(test_question)
        finally:
            if agent is not None:
                agent.close()
            GraphExploreSignature.__doc__ = original_doc
    except Exception as exc:
        print(f"  [retest] Erreur : {type(exc).__name__}: {exc}")
        return None


def apply_prompt_to_agent(new_prompt: str) -> bool:
    agent_file = ROOT / "graph_agent" / "agent.py"
    if not agent_file.exists():
        print(f"  [apply] {agent_file} introuvable")
        return False

    content = agent_file.read_text(encoding="utf-8")
    start_marker = 'class GraphExploreSignature(dspy.Signature):\n    """'
    end_marker = '    """\n    question'

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        print("  [apply] Impossible de localiser GraphExploreSignature.__doc__ dans agent.py")
        return False

    indented = "\n".join(("    " + line if line.strip() else "") for line in new_prompt.strip().splitlines())
    before = content[: start_idx + len(start_marker)]
    after = content[end_idx:]
    agent_file.write_text(before + "\n" + indented + "\n" + after, encoding="utf-8")
    print("  ✓ GraphExploreSignature.__doc__ mis à jour dans agent.py")
    return True


def optimize(do_retest: bool = False, max_iterations: int = MAX_ITERATIONS) -> None:
    print("═" * 70)
    print("  OPTIMISATION DU PROMPT GRAPHAGENT")
    print("═" * 70)

    print("\n1. Chargement du dataset...")
    items, _ = load_history()
    if not items:
        print("  Aucun item chargé. Lance d'abord demo_client.3_juges_iteratif.")
        return

    print("\n2. Chargement des juges...")
    judge_prompts = load_judge_prompts()
    if not judge_prompts:
        print("  Aucun juge disponible.")
        return

    print("\n3. Chargement du prompt de l'agent...")
    ensure_agent_prompt_registered()
    agent_prompt, current_version = load_agent_prompt()
    print(f"  Prompt v{current_version} chargé ({len(agent_prompt)} chars)")

    mlflow.set_experiment(AGENT_EXPERIMENT_NAME)
    validation_question = (
        "Trace the complete data flow when a customer submits an order through the portal. "
        "Follow the execution path from the HTTP entry point through all business logic layers to final data persistence."
    )

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 70}")
        print(f"  ITÉRATION {iteration}/{max_iterations}")
        print(f"{'=' * 70}")

        with mlflow.start_run(run_name=f"optimize_iter_{iteration}"):
            mlflow.log_params(
                {
                    "iteration": iteration,
                    "n_items": len(items),
                    "agent_prompt_v": current_version,
                    "judge_model": JUDGE_MODEL,
                    "judge_api_base": REMOTE_API_BASE,
                }
            )
            mlflow.log_text(agent_prompt, f"agent_prompt_iter_{iteration}.txt")

            all_verdicts: list[str] = []
            feedbacks_to_fix: list[dict[str, str]] = []

            for index, item in enumerate(items, start=1):
                print(f"\n  Évaluation {index}/{len(items)}: {item['question'][:80]}...")
                verdicts = evaluate_item(judge_prompts, item)
                for judge_name, verdict_data in verdicts.items():
                    verdict_label = verdict_data["verdict_label"]
                    all_verdicts.append(verdict_label)
                    print(f"    {judge_name}: {verdict_label}")
                    if verdict_label != "SATISFAISANT":
                        feedbacks_to_fix.append(
                            {
                                "judge": judge_name,
                                "question": item["question"],
                                "answer": item["answer"],
                                "verdict": verdict_label,
                                "recommendation": verdict_data["recommendation"],
                            }
                        )

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

            print(f"\n  Raffinement du prompt ({len(feedbacks_to_fix)} problème(s))...")
            new_prompt = refine_agent_prompt(agent_prompt, feedbacks_to_fix)
            commit_message = f"Iter {iteration}: {score:.0%} SATISFAISANT → raffinement"
            new_version = save_agent_prompt(new_prompt, commit_message, iteration)
            print(f"  ✓ Nouvelle version : {AGENT_PROMPT_NAME} v{new_version}")
            mlflow.log_metric("new_prompt_version", new_version, step=iteration)

            agent_prompt = new_prompt
            current_version = new_version

            if do_retest:
                print("\n  Re-test de l'agent sur la question de validation...")
                retest_result = retest_agent(new_prompt, validation_question)
                if retest_result:
                    n_visited = len(retest_result.get("visited", []))
                    print(f"  ✓ Re-test OK — {n_visited} noeud(s) visité(s)")
                    mlflow.log_metric("retest_n_visited", n_visited, step=iteration)
                else:
                    print("  ✗ Re-test échoué (agent ou Neo4j non disponible)")
    else:
        print(f"\n⚠️  Max itérations ({max_iterations}) atteint sans convergence complète.")

    print(f"\n{'─' * 70}")
    print("  APPLICATION DU PROMPT OPTIMISÉ DANS agent.py")
    print(f"{'─' * 70}")
    if apply_prompt_to_agent(agent_prompt):
        print(f"  Prompt v{current_version} appliqué. Relance l'agent pour valider.")
    else:
        print(f"  Copie manuelle : prompts/{AGENT_PROMPT_NAME}/{current_version}")

    print("\n✅ Optimisation terminée.")


def main(do_retest: bool = False, max_iter: int = MAX_ITERATIONS) -> None:
    optimize(do_retest=do_retest, max_iterations=max_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise le prompt GraphExploreSignature pour mall")
    parser.add_argument("--retest", action="store_true", help="Re-run l'agent à chaque itération")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Nb max d'itérations")
    args = parser.parse_args()
    main(do_retest=args.retest, max_iter=args.max_iter)
