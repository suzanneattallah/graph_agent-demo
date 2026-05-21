"""
Boucle itérative d'amélioration des juges MLflow pour la démo mall.

Usage :
  python -m demo_client.3_juges_iteratif
  python demo_client\3_juges_iteratif.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlflow
import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import (
        AGENT_EXPERIMENT,
        API_KEY,
        ARTIFACTS_DIR,
        JUDGES,
        JUDGE_EXPERIMENT,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        MLFLOW_URL,
        REFINER_MODEL,
        REMOTE_API_BASE,
    )
except ImportError:
    from demo_client.config import (
        AGENT_EXPERIMENT,
        API_KEY,
        ARTIFACTS_DIR,
        JUDGES,
        JUDGE_EXPERIMENT,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        MLFLOW_URL,
        REFINER_MODEL,
        REMOTE_API_BASE,
    )

from graph_agent.juges.register_juges import FALLBACK_PROMPTS

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = ARTIFACTS_DIR / ".judge_prompts_cache.json"
SCORE_BY_VERDICT = {"SATISFAISANT": 1.0, "À AMÉLIORER": 0.5, "INSUFFISANT": 0.0}

mlflow.set_tracking_uri(MLFLOW_URL)

# Pas de client global : on recrée un client frais à chaque appel pour éviter
# les coupures de connexion TCP (keep-alive VPN / idle timeout serveur).
def _make_client(timeout: float = 180.0) -> OpenAI:
    return OpenAI(base_url=REMOTE_API_BASE, api_key=API_KEY, timeout=timeout)


def _llm_call(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
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
            time.sleep(retry_delay)
    raise RuntimeError(f"LLM échoué après {max_retries} tentatives : {last_error}")


def _normalize(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode().upper()


def extract_verdict(text: str) -> str:
    normalized = _normalize(text)
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*SATISFAISANT", normalized):
        return "SATISFAISANT"
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT|NON SATISFAISANT)", normalized):
        match = re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT|NON SATISFAISANT)", normalized)
        label = match.group(1) if match else ""
        return "INSUFFISANT" if "INSUFFISANT" in label else "À AMÉLIORER"
    if re.search(r"(DECISION|EVALUATION|RESULTAT)\s*:?\s*[\*#]*\s*SATISFAISANT", normalized):
        return "SATISFAISANT"
    if "SATISFAISANT" in normalized:
        return "SATISFAISANT"
    if "A AMELIORER" in normalized or "INSUFFISANT" in normalized:
        return "INSUFFISANT" if "INSUFFISANT" in normalized else "À AMÉLIORER"
    return "INCONNU"


def extract_score(text: str) -> float:
    """Extrait le score numérique (0.0/0.5/1.0) directement depuis la ligne SCORE du juge."""
    match = re.search(r"SCORE\s*:?\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Fallback : déduit depuis le verdict
    v = extract_verdict(text)
    return {"SATISFAISANT": 1.0, "À AMÉLIORER": 0.5, "INSUFFISANT": 0.0}.get(v, 0.0)


def extract_numeric_metric(text: str, metric_name: str) -> int | None:
    """Extrait une métrique numérique comme N_VISITED=[7], N_VISITED=7 ou N_VISITED: 7."""
    match = re.search(rf"{metric_name}\s*[=:]\s*\[?(\d+)\]?", text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def extract_recommendation(text: str) -> str:
    match = re.search(r"RECOMMANDATION\s*:?\s*(.*?)(?:\n{2,}|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:600]
    return text[:400]


def _read_artifact(mlf_client: mlflow.MlflowClient, run_id: str, name: str) -> str:
    try:
        response = requests.get(
            f"{MLFLOW_URL}/get-artifact",
            params={"run_uuid": run_id, "path": name},
            timeout=10,
        )
        if response.status_code == 200:
            return response.text
    except Exception:
        pass

    try:
        artifact_path = mlf_client.download_artifacts(run_id, name)
        return Path(artifact_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_metric(run: dict, key: str) -> float:
    value = run.get(f"metrics.{key}")
    if value is None:
        return 0.0
    try:
        import math

        return 0.0 if math.isnan(value) else float(value)
    except Exception:
        return float(value)


def fetch_agent_traces(limit: int = 7) -> list[dict[str, object]]:
    experiment = mlflow.get_experiment_by_name(AGENT_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{AGENT_EXPERIMENT}' introuvable. Lance d'abord : python -m demo_client.2_generer_traces"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=limit,
        order_by=["start_time DESC"],
    )

    mlf_client = mlflow.MlflowClient()
    traces: list[dict[str, object]] = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id[:8])
        question = run.get("params.question", "").strip()
        answer = _read_artifact(mlf_client, run_id, "answer.txt").strip()
        visited_nodes = _read_artifact(mlf_client, run_id, "visited_nodes.txt").strip()
        notes = _read_artifact(mlf_client, run_id, "notes.txt").strip()
        nav_path = _read_artifact(mlf_client, run_id, "navigation_path.txt").strip()

        if not question or not answer:
            print(f"  [skip] Run {run_name} : question ou réponse manquante")
            continue

        traces.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "question": question,
                "answer": answer,
                "visited_nodes": visited_nodes,
                "notes": notes,
                "nav_path": nav_path,
                "n_visited": int(_safe_metric(run, "n_visited_nodes")),
                "n_notes": int(_safe_metric(run, "n_notes")),
                "elapsed": _safe_metric(run, "elapsed_seconds"),
            }
        )

    print(f"  {len(traces)} trace(s) valide(s) récupérée(s)")
    return traces


def build_trace_info(trace: dict[str, object]) -> str:
    return "\n".join(
        [
            f"Noeuds visités ({trace['n_visited']}) :",
            str(trace["visited_nodes"] or "(aucun noeud visité)"),
            "",
            f"Chemin de navigation : {trace['nav_path'] or '--'}",
            "",
            f"Notes de l'agent ({trace['n_notes']}) :",
            str(trace["notes"] or "(aucune note enregistrée)"),
            "",
            f"Durée : {float(trace['elapsed']):.1f}s",
        ]
    )


def load_judge_prompts() -> dict[str, str]:
    mlf_client = mlflow.MlflowClient()
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    prompts: dict[str, str] = {}

    for judge_name in JUDGES:
        registry_name = f"{JUDGE_PREFIX}-{judge_name}"
        try:
            versions = mlf_client.search_prompt_versions(name=registry_name)
            if versions:
                latest_version = max(int(version.version) for version in versions)
                prompt_obj = mlflow.genai.load_prompt(f"prompts:/{registry_name}/{latest_version}")
                prompts[judge_name] = prompt_obj.template
                print(f"  ✓ {judge_name} v{latest_version} chargé depuis le Registry")
                continue
        except Exception as exc:
            print(f"  [warn] {judge_name} : Registry inaccessible ({exc})")

        if judge_name in cache:
            print(f"  [cache] {judge_name} : utilisation du cache local")
            prompts[judge_name] = cache[judge_name]
        else:
            print(f"  [fallback] {judge_name} : utilisation du prompt initial")
            prompts[judge_name] = FALLBACK_PROMPTS[judge_name]

    return prompts


def register_prompt_version(name: str, template: str, commit_message: str, tags: dict[str, str]) -> str | None:
    try:
        prompt_version = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags,
        )
        return f"prompts:/{name}/{prompt_version.version}"
    except Exception as exc:
        print(f"  [Registry] Échec '{name}' : {type(exc).__name__}: {exc}")
        return None


def run_judge(judge_instructions: str, question: str, answer: str, trace_info: str) -> str:
    filled_prompt = (
        judge_instructions.replace("{{ inputs }}", question)
        .replace("{{ outputs }}", answer)
        .replace("{{ trace }}", trace_info)
    )
    # System message "/no_think" désactive le mode thinking de Qwen3
    # et force des réponses concises dans le format attendu.
    system_msg = (
        "/no_think\n"
        "You are a strict code quality judge. "
        "Respond ONLY in the exact format: MÉTRIQUES MESURÉES / ANALYSE / VERDICT / SCORE / RECOMMANDATION. "
        "Be concise. Do not explain your reasoning process. Output the verdict directly."
    )
    try:
        return _llm_call(
            JUDGE_MODEL,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": filled_prompt},
            ],
            timeout=300.0,
        )
    except Exception as exc:
        return f"[ERREUR JUGE] {type(exc).__name__}: {str(exc)[:200]}"


def refine_judge_prompt(judge_name: str, current_prompt: str, disagreements: list[dict[str, str]]) -> str:
    disagreement_text = "\n\n".join(
        [
            f"--- Cas {index + 1} ---\n"
            f"Question : {item['question']}\n"
            f"Réponse de l'agent (début) : {item['answer'][:400]}...\n"
            f"Verdict du juge : {item['judge_verdict']}\n"
            f"Feedback humain : {item['human_feedback']}"
            for index, item in enumerate(disagreements)
        ]
    )

    system_msg = (
        f"Tu es un expert en prompt engineering pour des juges LLM. "
        f"Tu dois améliorer le prompt du juge '{judge_name}' pour un agent "
        f"qui explore un graphe de code source Neo4j du projet mall Java e-commerce. "
        f"CONTRAINTES À PRÉSERVER ABSOLUMENT : "
        f"(1) Structure en 3 étapes : ÉTAPE 1 (extraction métriques observables) / "
        f"    ÉTAPE 2 (grille d'évaluation mécanique) / ÉTAPE 3 (présomption d'innocence). "
        f"(2) Format de sortie : MÉTRIQUES MESURÉES / ANALYSE / VERDICT / SCORE / RECOMMANDATION. "
        f"(3) Verdicts : SATISFAISANT / À AMÉLIORER / INSUFFISANT. "
        f"(4) PRÉSOMPTION D'INNOCENCE STRICTE : défaut = SATISFAISANT. "
        f"    RÈGLE D'OR : en cas d'hésitation → SATISFAISANT. "
        f"(5) SCORE numérique : 1.0=SATISFAISANT, 0.5=À AMÉLIORER, 0.0=INSUFFISANT. "
        f"(6) Variables {{{{ inputs }}}}, {{{{ outputs }}}}, {{{{ trace }}}} intactes. "
        f"(7) Les métriques doivent rester OBJECTIVES et OBSERVABLES (pas d'opinion). "
        f"Retourne UNIQUEMENT le nouveau prompt complet, sans explication."
    )
    user_msg = (
        f"Prompt actuel du juge :\n\n{current_prompt}\n\n"
        f"Feedbacks humains à intégrer :\n\n{disagreement_text}\n\n"
        f"Réécris le prompt pour intégrer ces retours dans les futurs jugements."
    )

    try:
        return _llm_call(
            REFINER_MODEL,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            timeout=240.0,
        )
    except Exception as exc:
        print(f"  [raffinement] Échec pour {judge_name} : {exc} → prompt inchangé")
        return current_prompt



def _log_score_metrics(full_history: list[dict[str, object]], iteration: int) -> tuple[dict[str, float], float]:
    """Log les scores numériques par juge + métriques d'exploration dans MLflow."""
    judge_scores: dict[str, float] = {}
    all_scores: list[float] = []

    for judge_name in JUDGES:
        scores = [
            record["judges"][judge_name]["score"]
            for record in full_history
            if judge_name in record["judges"]
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        sat_pct   = sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0.0
        judge_scores[judge_name] = avg_score
        mlflow.log_metric(f"score_{judge_name}",        avg_score, step=iteration)
        mlflow.log_metric(f"satisfaisant_{judge_name}", sat_pct,   step=iteration)
        all_scores.extend(scores)

    global_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    mlflow.log_metric("score_global", global_score, step=iteration)

    # Métriques d'exploration extraites par JugeExploration (n_visited, n_layers)
    for record in full_history:
        if "JugeExploration" in record["judges"]:
            raw = record["judges"]["JugeExploration"]["verdict"]
            nv = extract_numeric_metric(raw, "N_VISITED")
            nl = extract_numeric_metric(raw, "N_LAYERS")
            if nv is not None:
                mlflow.log_metric("n_visited_avg", nv, step=iteration)
            if nl is not None:
                mlflow.log_metric("n_layers_avg",  nl, step=iteration)
            break

    return judge_scores, global_score


def run_iteration(
    traces: list[dict[str, object]],
    current_prompts: dict[str, str],
    iteration: int,
    cache: dict[str, str],
) -> tuple[int, list[dict[str, object]]]:
    print(f"\n{'=' * 70}")
    print(f"  ITÉRATION {iteration}")
    print(f"{'=' * 70}")

    judge_disagreements: dict[str, list[dict[str, str]]] = {judge_name: [] for judge_name in JUDGES}
    full_history: list[dict[str, object]] = []

    for idx, trace in enumerate(traces, start=1):
        print(f"\n--- Trace {idx}/{len(traces)} : {str(trace['run_name'])[:50]} ---")
        print(f"Question : {str(trace['question'])[:150]}")
        print(f"Réponse  : {str(trace['answer'])[:250]}...")
        print(f"Noeuds visités : {trace['n_visited']} | Durée : {float(trace['elapsed']):.1f}s\n")

        trace_info = build_trace_info(trace)
        trace_record: dict[str, object] = {
            "run_id": trace["run_id"],
            "question": trace["question"],
            "answer": trace["answer"],
            "judges": {},
        }

        for judge_name in JUDGES:
            print(f"\n>> {judge_name}")
            verdict = run_judge(current_prompts[judge_name], str(trace["question"]), str(trace["answer"]), trace_info)
            print(verdict)

            human_feedback = input("\nTon feedback sur ce verdict (texte libre, ou 'ok' si d'accord) : ").strip()
            trace_record["judges"][judge_name] = {
                "verdict": verdict,
                "verdict_label": extract_verdict(verdict),
                "score": extract_score(verdict),
                "human_feedback": human_feedback,
            }

            if human_feedback.lower() not in ("ok", "", "ras", "accord"):
                judge_disagreements[judge_name].append(
                    {
                        "question": str(trace["question"]),
                        "answer": str(trace["answer"]),
                        "judge_verdict": verdict,
                        "human_feedback": human_feedback,
                    }
                )

        full_history.append(trace_record)

    history_file = ARTIFACTS_DIR / f"history_iteration_{iteration}.json"
    history_file.write_text(json.dumps(full_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Historique sauvegardé : {history_file}")

    md_lines = [f"# Itération {iteration} — Verdicts demo_client\n"]
    for record in full_history:
        md_lines.append(f"## Run {str(record['run_id'])[:8]}\n")
        md_lines.append(f"**Question :** {record['question']}\n")
        for judge_name, judge_data in record["judges"].items():
            verdict_label = extract_verdict(judge_data["verdict"])
            md_lines.append(f"### {judge_name} → {verdict_label}\n")
            md_lines.append(f"```\n{judge_data['verdict'][:600]}\n```\n")
            md_lines.append(f"*Feedback humain :* {judge_data['human_feedback'] or '_(accord)_'}\n")
        md_lines.append("---\n")
    mlflow.log_text("\n".join(md_lines), f"verdicts_iteration_{iteration}.md")
    mlflow.log_text(json.dumps(full_history, ensure_ascii=False, indent=2), f"history_iteration_{iteration}.json")

    print(f"\n{'─' * 70}")
    print("  RAFFINEMENT DES JUGES")
    print(f"{'─' * 70}")

    n_refined = 0
    for judge_name, disagreements in judge_disagreements.items():
        if not disagreements:
            print(f"\n>> {judge_name} : aucun désaccord, prompt inchangé.")
            continue

        print(f"\n>> {judge_name} : {len(disagreements)} désaccord(s) → raffinement...")
        new_prompt = refine_judge_prompt(judge_name, current_prompts[judge_name], disagreements)
        feedback_summary = " | ".join(
            disagreement["human_feedback"][:80] for disagreement in disagreements if disagreement["human_feedback"]
        )[:300]
        commit_message = f"Iter {iteration}: {feedback_summary or 'raffinement automatique'}"

        registry_name = f"{JUDGE_PREFIX}-{judge_name}"
        prompt_uri = register_prompt_version(
            name=registry_name,
            template=new_prompt,
            commit_message=commit_message,
            tags={"iteration": str(iteration), "judge_type": judge_name},
        )
        if prompt_uri:
            print(f"  [Registry] {prompt_uri}")
            current_prompts[judge_name] = new_prompt
            cache[judge_name] = new_prompt
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            n_refined += 1

    total_disagreements = sum(len(disagreements) for disagreements in judge_disagreements.values())
    verdicts_all = [extract_verdict(record["judges"][judge_name]["verdict"]) for record in full_history for judge_name in JUDGES]
    satisfaisant_pct = verdicts_all.count("SATISFAISANT") / len(verdicts_all) if verdicts_all else 0.0
    judge_scores, global_score = _log_score_metrics(full_history, iteration)

    mlflow.log_metric("total_disagreements", total_disagreements, step=iteration)
    mlflow.log_metric("judges_refined", n_refined, step=iteration)
    mlflow.log_metric("satisfaisant_pct", satisfaisant_pct, step=iteration)
    for judge_name, disagreements in judge_disagreements.items():
        mlflow.log_metric(f"{judge_name}_disagreements", len(disagreements), step=iteration)

    for judge_name, score in judge_scores.items():
        print(f"  score_{judge_name}: {score:.2f}")
    print(f"  score_global: {global_score:.2f}")
    print(
        f"\nItération {iteration} : {n_refined} juge(s) raffiné(s), {total_disagreements} désaccord(s), "
        f"{satisfaisant_pct:.0%} SATISFAISANT."
    )
    return n_refined, full_history


def main(limit: int = 5, iterations: int = 3) -> None:
    print(f"Récupération des traces depuis '{AGENT_EXPERIMENT}'...")
    traces = fetch_agent_traces(limit=limit)
    if not traces:
        raise SystemExit("Aucune trace trouvée. Lance d'abord demo_client.2_generer_traces.")

    print(f"\n{len(traces)} trace(s) prête(s) à évaluer.\n")
    print("Chargement des prompts de juges...")
    current_prompts = load_judge_prompts()
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}

    mlflow.set_experiment(JUDGE_EXPERIMENT)
    for iteration in range(1, iterations + 1):
        with mlflow.start_run(run_name=f"iteration_{iteration}"):
            mlflow.log_params(
                {
                    "iteration": iteration,
                    "n_traces": len(traces),
                    "judge_model": JUDGE_MODEL,
                    "judge_api_base": REMOTE_API_BASE,
                }
            )
            n_refined, _ = run_iteration(traces, current_prompts, iteration, cache)

        if n_refined == 0:
            print(f"\n✅ Convergence atteinte à l'itération {iteration} — aucun juge raffiné.")
            break

    print("\nLance ensuite : python -m demo_client.4_optimize_agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boucle itérative de raffinement des juges pour mall")
    parser.add_argument("--limit", type=int, default=5, help="Nombre de traces à évaluer")
    parser.add_argument("--iterations", type=int, default=3, help="Nombre d'itérations")
    args = parser.parse_args()
    main(limit=args.limit, iterations=args.iterations)
