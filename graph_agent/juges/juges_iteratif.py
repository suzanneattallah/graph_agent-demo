"""
Boucle itérative d'amélioration des juges MLflow pour GraphAgent.

Workflow :
1. Récupère les traces de l'agent depuis MLflow (experiment "graph-agent")
2. Pour chaque trace, fait tourner les 4 juges via Ollama (verdict + analyse + recommandation)
3. Tu donnes ton feedback humain sémantique (texte libre) sur chaque verdict
4. Ollama raffine les instructions de chaque juge à partir des désaccords
5. Met à jour les juges dans le MLflow Prompt Registry
6. Sauvegarde l'historique pour Optimize_Agent_Iteratif.py

Usage : python -m graph_agent.juges.juges_iteratif
"""

import json
import re
import time
import requests
import mlflow
from openai import OpenAI
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URL              = "http://localhost:5000"
AGENT_EXPERIMENT_NAME   = "graph-agent"
JUDGE_TRACKING_EXP      = "GraphAgent_Juges_Iteratif"
JUDGE_PREFIX            = "graph-agent"

# Endpoint Mac (modèle puissant) — fallback Ollama local si non disponible
JUDGE_MODEL    = "qwen3.6-27b"
JUDGE_BASE_URL = "http://CHLASLITASSAPR1.lan.la.sqli.com:8080/v1"
JUDGE_API_KEY  = "none"

# Même endpoint pour le raffinement (qualité maximale pour l'amélioration des prompts)
REFINER_MODEL    = "qwen3.6-27b"
REFINER_BASE_URL = "http://CHLASLITASSAPR1.lan.la.sqli.com:8080/v1"

JUDGES = [
    "JugeExploration",
    "JugePrecisionTechnique",
    "JugeRaisonnement",
    "JugeAmeliorations",
]

ARTIFACTS_DIR = Path(__file__).parent / "artifact"
ARTIFACTS_DIR.mkdir(exist_ok=True)
CACHE_FILE = Path(__file__).parent / ".judge_prompts_cache.json"

# ── Fallback prompts (si le Registry MLflow n'est pas disponible) ─────────────
# Importer depuis register_juges pour éviter la duplication
try:
    from .register_juges import FALLBACK_PROMPTS
except ImportError:
    from register_juges import FALLBACK_PROMPTS

mlflow.set_tracking_uri(MLFLOW_URL)

def _make_judge_client(timeout: float = 180.0) -> OpenAI:
    return OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=timeout)

def _make_refiner_client(timeout: float = 240.0) -> OpenAI:
    return OpenAI(base_url=REFINER_BASE_URL, api_key=JUDGE_API_KEY, timeout=timeout)


# ── LLM call avec retry ───────────────────────────────────────────────────────
def _llm_call(client_factory, model: str, messages: list, temperature: float = 0.0,
              max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """Appel LLM avec retry automatique et streaming. client_factory est un callable sans argument."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        content = ""
        try:
            client = client_factory()
            for chunk in client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            ):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content
            if content.strip():
                return content.strip()
            last_error = "réponse vide"
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt}/{max_retries}] {type(e).__name__}: {str(e)[:120]}")
        if attempt < max_retries:
            time.sleep(retry_delay)
    raise RuntimeError(f"LLM échoué après {max_retries} tentatives : {last_error}")


# ── Extraction verdict ────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().upper()


def extract_verdict(text: str) -> str:
    t = _normalize(text)
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*SATISFAISANT", t):
        return "SATISFAISANT"
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT|NON SATISFAISANT)", t):
        m = re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT|NON SATISFAISANT)", t)
        label = m.group(1) if m else ""
        return "INSUFFISANT" if "INSUFFISANT" in label else "À AMÉLIORER"
    if re.search(r"(DECISION|EVALUATION|RESULTAT)\s*:?\s*[\*#]*\s*SATISFAISANT", t):
        return "SATISFAISANT"
    if "SATISFAISANT" in t:
        return "SATISFAISANT"
    if "A AMELIORER" in t or "INSUFFISANT" in t:
        return "INSUFFISANT" if "INSUFFISANT" in t else "À AMÉLIORER"
    return "INCONNU"


def extract_score(text: str) -> float:
    """Extrait le score numérique (0.0 / 0.5 / 1.0) de la réponse d'un juge.
    Fallback sur le verdict si SCORE absent."""
    m = re.search(r"SCORE\s*:?\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fallback : déduction depuis le verdict
    v = extract_verdict(text)
    return {"SATISFAISANT": 1.0, "À AMÉLIORER": 0.5, "INSUFFISANT": 0.0}.get(v, 0.0)


def extract_numeric_metric(text: str, metric_name: str) -> int | None:
    """Extrait une métrique numérique comme N_VISITED=[7], N_VISITED=7 ou N_VISITED: 7."""
    m = re.search(rf"{metric_name}\s*[=:]\s*\[?(\d+)\]?", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None



def extract_recommendation(text: str) -> str:
    match = re.search(r"RECOMMANDATION\s*:?\s*(.*?)(?:\n{2,}|$)", text,
                      re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:600]
    return text[:400]


# ── MLflow : lecture des traces de l'agent ───────────────────────────────────
def _read_artifact(mlf_client, run_id: str, name: str) -> str:
    """Lit un artifact texte via API REST MLflow (fiable avec Podman)."""
    try:
        resp = requests.get(
            f"{MLFLOW_URL}/get-artifact",
            params={"run_uuid": run_id, "path": name},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    try:
        path = mlf_client.download_artifacts(run_id, name)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _safe_metric(run, key):
    val = run.get(f"metrics.{key}")
    if val is None:
        return 0
    try:
        import math
        return 0 if math.isnan(val) else val
    except Exception:
        return val


def fetch_agent_traces(limit: int = 7) -> list[dict]:
    """Récupère les N dernières traces depuis l'experiment graph-agent."""
    exp = mlflow.get_experiment_by_name(AGENT_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(
            f"Experiment '{AGENT_EXPERIMENT_NAME}' introuvable. "
            "Lance d'abord : python -m graph_agent.juges.generer_traces"
        )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=limit,
        order_by=["start_time DESC"],
    )

    mlf_client = mlflow.MlflowClient()
    traces = []
    for _, run in runs.iterrows():
        run_id   = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id[:8])

        answer        = _read_artifact(mlf_client, run_id, "answer.txt")
        visited_nodes = _read_artifact(mlf_client, run_id, "visited_nodes.txt")
        notes         = _read_artifact(mlf_client, run_id, "notes.txt")
        nav_path      = _read_artifact(mlf_client, run_id, "navigation_path.txt")
        question      = run.get("params.question", "").strip()

        if not question or not answer:
            print(f"  [skip] Run {run_name} : artifacts question/answer manquants")
            continue

        traces.append({
            "run_id":         run_id,
            "run_name":       run_name,
            "question":       question,
            "answer":         answer.strip(),
            "visited_nodes":  visited_nodes.strip(),
            "notes":          notes.strip(),
            "nav_path":       nav_path.strip(),
            "n_visited":      int(_safe_metric(run, "n_visited_nodes")),
            "n_notes":        int(_safe_metric(run, "n_notes")),
            "elapsed":        _safe_metric(run, "elapsed_seconds"),
        })

    print(f"  {len(traces)} trace(s) valide(s) récupérée(s)")
    return traces


def build_trace_info(trace: dict) -> str:
    """Construit la chaîne trace_info passée aux juges."""
    lines = [
        f"Noeuds visités ({trace['n_visited']}) :",
        trace["visited_nodes"] or "(aucun noeud visité)",
        "",
        f"Chemin de navigation : {trace['nav_path'] or '--'}",
        "",
        f"Notes de l'agent ({trace['n_notes']}) :",
        trace["notes"] or "(aucune note enregistrée)",
        "",
        f"Durée : {trace['elapsed']:.1f}s",
    ]
    return "\n".join(lines)


# ── Prompt Registry : lecture + écriture ─────────────────────────────────────
def load_judge_prompts() -> dict[str, str]:
    """Charge les prompts depuis le Registry, avec fallback sur le cache local."""
    mlf_client = mlflow.MlflowClient()
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    prompts = {}

    for j in JUDGES:
        registry_name = f"{JUDGE_PREFIX}-{j}"
        try:
            versions = mlf_client.search_prompt_versions(name=registry_name)
            if versions:
                latest_v = max(int(v.version) for v in versions)
                prompt_obj = mlflow.genai.load_prompt(f"prompts:/{registry_name}/{latest_v}")
                prompts[j] = prompt_obj.template
                print(f"  ✓ {j} v{latest_v} chargé depuis le Registry")
                continue
        except Exception as e:
            print(f"  [warn] {j} : Registry inaccessible ({e})")

        if j in cache:
            print(f"  [cache] {j} : utilisation du cache local")
            prompts[j] = cache[j]
        else:
            print(f"  [fallback] {j} : utilisation du prompt initial")
            prompts[j] = FALLBACK_PROMPTS[j]

    return prompts


def register_prompt_version(name: str, template: str, commit_msg: str, tags: dict) -> str | None:
    try:
        pv = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_msg,
            tags=tags,
        )
        return f"prompts:/{name}/{pv.version}"
    except Exception as e:
        print(f"  [Registry] Échec '{name}' : {type(e).__name__}: {e}")
        return None


# ── Juge : exécution + raffinement ───────────────────────────────────────────
def run_judge(judge_instructions: str, question: str, answer: str, trace_info: str) -> str:
    filled = (
        judge_instructions
        .replace("{{ inputs }}", question)
        .replace("{{ outputs }}", answer)
        .replace("{{ trace }}", trace_info)
    )
    try:
        return _llm_call(
            _make_judge_client,
            JUDGE_MODEL,
            [{"role": "user", "content": filled}],
        )
    except Exception as e:
        return f"[ERREUR JUGE] {type(e).__name__}: {str(e)[:200]}"


def refine_judge_prompt(judge_name: str, current_prompt: str, disagreements: list[dict]) -> str:
    """Réécrit le prompt d'un juge à partir des feedbacks humains via Ollama."""
    disagreement_text = "\n\n".join([
        f"--- Cas {i+1} ---\n"
        f"Question : {d['question']}\n"
        f"Réponse de l'agent (début) : {d['answer'][:400]}...\n"
        f"Verdict du juge : {d['judge_verdict']}\n"
        f"Feedback humain : {d['human_feedback']}"
        for i, d in enumerate(disagreements)
    ])

    system_msg = (
        f"Tu es un expert en prompt engineering pour des juges LLM. "
        f"Tu dois améliorer le prompt du juge '{judge_name}' pour un agent "
        f"qui explore un graphe de code source Neo4j (projet mall Java e-commerce). "
        f"CONTRAINTES À PRÉSERVER ABSOLUMENT dans le nouveau prompt : "
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
            _make_refiner_client,
            REFINER_MODEL,
            [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
        )
    except Exception as e:
        print(f"  [raffinement] Échec pour {judge_name} : {e} → prompt inchangé")
        return current_prompt


# ── Boucle itérative principale ───────────────────────────────────────────────
def run_iteration(traces: list[dict], current_prompts: dict[str, str],
                  iteration: int, cache: dict) -> tuple[int, list[dict]]:
    """Une itération : juges → feedback humain → raffinement. Retourne (n_raffinés, full_history)."""
    print(f"\n{'='*70}")
    print(f"  ITÉRATION {iteration}")
    print(f"{'='*70}")

    judge_disagreements = {j: [] for j in JUDGES}
    full_history = []

    for idx, trace in enumerate(traces):
        print(f"\n--- Trace {idx+1}/{len(traces)} : {trace['run_name'][:50]} ---")
        print(f"Question : {trace['question'][:150]}")
        print(f"Réponse  : {trace['answer'][:250]}...")
        print(f"Noeuds visités : {trace['n_visited']} | Durée : {trace['elapsed']:.1f}s\n")

        trace_info   = build_trace_info(trace)
        trace_record = {
            "run_id":   trace["run_id"],
            "question": trace["question"],
            "answer":   trace["answer"],
            "judges":   {},
        }

        for judge_name in JUDGES:
            print(f"\n>> {judge_name}")
            verdict = run_judge(
                current_prompts[judge_name],
                trace["question"],
                trace["answer"],
                trace_info,
            )
            print(verdict)

            human_feedback = input(
                f"\nTon feedback sur ce verdict (texte libre, ou 'ok' si d'accord) : "
            ).strip()

            trace_record["judges"][judge_name] = {
                "verdict":        verdict,
                "verdict_label":  extract_verdict(verdict),
                "score":          extract_score(verdict),
                "human_feedback": human_feedback,
            }

            if human_feedback.lower() not in ("ok", "", "ras", "accord"):
                judge_disagreements[judge_name].append({
                    "question":      trace["question"],
                    "answer":        trace["answer"],
                    "judge_verdict": verdict,
                    "human_feedback": human_feedback,
                })

        full_history.append(trace_record)

    # Sauvegarde JSON pour optimize_agent.py
    history_file = ARTIFACTS_DIR / f"history_iteration_{iteration}.json"
    history_file.write_text(json.dumps(full_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Historique sauvegardé : {history_file}")

    # Résumé markdown dans MLflow
    md_lines = [f"# Itération {iteration} — Verdicts GraphAgent\n"]
    for rec in full_history:
        md_lines.append(f"## Run {rec['run_id'][:8]}\n")
        md_lines.append(f"**Question :** {rec['question']}\n")
        for jn, jd in rec["judges"].items():
            v = extract_verdict(jd["verdict"])
            md_lines.append(f"### {jn} → {v}\n")
            md_lines.append(f"```\n{jd['verdict'][:600]}\n```\n")
            md_lines.append(f"*Feedback humain :* {jd['human_feedback'] or '_(accord)_'}\n")
        md_lines.append("---\n")
    mlflow.log_text("\n".join(md_lines), f"verdicts_iteration_{iteration}.md")
    mlflow.log_text(
        json.dumps(full_history, ensure_ascii=False, indent=2),
        f"history_iteration_{iteration}.json",
    )

    # ── Raffinement des juges avec désaccords ─────────────────────────────────
    print(f"\n{'─'*70}")
    print("  RAFFINEMENT DES JUGES")
    print(f"{'─'*70}")

    n_refined = 0
    for judge_name, disagreements in judge_disagreements.items():
        if not disagreements:
            print(f"\n>> {judge_name} : aucun désaccord, prompt inchangé.")
            continue

        print(f"\n>> {judge_name} : {len(disagreements)} désaccord(s) → raffinement...")
        new_prompt = refine_judge_prompt(judge_name, current_prompts[judge_name], disagreements)

        feedback_summary = " | ".join(
            d["human_feedback"][:80] for d in disagreements if d["human_feedback"]
        )[:300]
        commit_msg = f"Iter {iteration}: {feedback_summary or 'raffinement automatique'}"

        registry_name = f"{JUDGE_PREFIX}-{judge_name}"
        uri = register_prompt_version(
            name=registry_name,
            template=new_prompt,
            commit_message=commit_msg,
            tags={"iteration": str(iteration), "judge_type": judge_name},
        )
        if uri:
            print(f"  [Registry] {uri}")
            current_prompts[judge_name] = new_prompt
            cache[judge_name] = new_prompt
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            n_refined += 1

    # Métriques MLflow de suivi — objectives et numériques
    total_disagreements = sum(len(d) for d in judge_disagreements.values())
    verdicts_all = [rec["judges"][j]["verdict_label"]
                    for rec in full_history for j in JUDGES]
    scores_all   = [rec["judges"][j]["score"]
                    for rec in full_history for j in JUDGES]
    satisfaisant_pct = verdicts_all.count("SATISFAISANT") / len(verdicts_all) if verdicts_all else 0
    avg_score        = sum(scores_all) / len(scores_all) if scores_all else 0.0

    mlflow.log_metric("total_disagreements", total_disagreements, step=iteration)
    mlflow.log_metric("judges_refined",      n_refined,           step=iteration)
    mlflow.log_metric("satisfaisant_pct",    satisfaisant_pct,    step=iteration)
    mlflow.log_metric("avg_score_global",    avg_score,           step=iteration)

    # Score par juge (pour courbes d'évolution individuelles)
    for jn in JUDGES:
        j_scores  = [rec["judges"][jn]["score"] for rec in full_history if jn in rec["judges"]]
        j_verdicts = [rec["judges"][jn]["verdict_label"] for rec in full_history if jn in rec["judges"]]
        j_avg   = sum(j_scores) / len(j_scores) if j_scores else 0.0
        j_sat   = j_verdicts.count("SATISFAISANT") / len(j_verdicts) if j_verdicts else 0.0
        mlflow.log_metric(f"score_{jn}",        j_avg,                    step=iteration)
        mlflow.log_metric(f"satisfaisant_{jn}", j_sat,                    step=iteration)
        mlflow.log_metric(f"{jn}_disagreements", len(judge_disagreements[jn]), step=iteration)

    # Métriques d'exploration extraites par JugeExploration (n_visited, n_layers)
    for rec in full_history:
        if "JugeExploration" in rec["judges"]:
            raw = rec["judges"]["JugeExploration"]["verdict"]
            nv = extract_numeric_metric(raw, "N_VISITED")
            nl = extract_numeric_metric(raw, "N_LAYERS")
            if nv is not None:
                mlflow.log_metric("n_visited_avg", nv, step=iteration)
            if nl is not None:
                mlflow.log_metric("n_layers_avg",  nl, step=iteration)
            break  # on log juste la première trace comme indicateur

    print(f"\nItération {iteration} : {n_refined} juge(s) raffiné(s), "
          f"{total_disagreements} désaccord(s), {satisfaisant_pct:.0%} SATISFAISANT.")
    return n_refined, full_history


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boucle itérative de raffinement des juges GraphAgent")
    parser.add_argument("--limit",       type=int, default=5,  help="Nombre de traces à évaluer")
    parser.add_argument("--iterations",  type=int, default=3,  help="Nombre d'itérations")
    args = parser.parse_args()

    print(f"Récupération des traces depuis '{AGENT_EXPERIMENT_NAME}'...")
    traces = fetch_agent_traces(limit=args.limit)
    if not traces:
        print("Aucune trace trouvée. Lance d'abord generer_traces.py.")
        exit(1)

    print(f"\n{len(traces)} trace(s) prête(s) à évaluer.\n")

    print("Chargement des prompts de juges...")
    current_prompts = load_judge_prompts()
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}

    mlflow.set_experiment(JUDGE_TRACKING_EXP)

    for i in range(1, args.iterations + 1):
        with mlflow.start_run(run_name=f"iteration_{i}"):
            mlflow.log_params({"iteration": i, "n_traces": len(traces), "judge_model": JUDGE_MODEL})
            n_refined, _ = run_iteration(traces, current_prompts, i, cache)

        if n_refined == 0:
            print(f"\n✅ Convergence atteinte à l'itération {i} — aucun juge raffiné.")
            break

    print("\nLance ensuite : python -m graph_agent.juges.optimize_agent")
