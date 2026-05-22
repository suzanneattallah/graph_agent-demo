"""
Optimisation automatique des prompts de juges via GEPA + MLflow Prompt Registry.

Workflow :
  1. Charge le dernier history_iteration_N.json (verdicts + feedbacks humains)
  2. Pour chaque juge, construit (train_data) avec verdicts attendus déduits
     du feedback humain
  3. Lance GepaPromptOptimizer sur chaque prompt de juge
  4. Enregistre la nouvelle version dans le Prompt Registry MLflow
  5. Met à jour le juge dans MLflow UI (section Judges)

Usage :
  python -m demo_client.5_optimize_judges
  python demo_client\\5_optimize_judges.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time as _time
from pathlib import Path

# Force UTF-8 stdout/stderr pour éviter UnicodeEncodeError sur Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
from mlflow import MlflowClient
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
from mlflow.genai.scorers import scorer
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import (
        API_KEY,
        ARTIFACTS_DIR,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        JUDGES,
        MLFLOW_URL,
        REMOTE_API_BASE,
    )
except ImportError:
    from demo_client.config import (
        API_KEY,
        ARTIFACTS_DIR,
        JUDGE_MODEL,
        JUDGE_PREFIX,
        JUDGES,
        MLFLOW_URL,
        REMOTE_API_BASE,
    )

# ── Configuration GEPA ────────────────────────────────────────────────────────
# Pour utiliser le endpoint Mac comme reflection model, on passe par litellm
# avec OPENAI_API_BASE pointant sur le endpoint distant.
os.environ.setdefault("OPENAI_API_BASE", REMOTE_API_BASE)
os.environ.setdefault("OPENAI_API_KEY", API_KEY if API_KEY != "none" else "sk-none")

# Reflection model : MLflow GEPA format "openai:/<model>" utilise OPENAI_API_BASE
REFLECTION_MODEL = f"openai:/{JUDGE_MODEL}"

VERDICTS = ["SATISFAISANT", "À AMÉLIORER", "INSUFFISANT"]
EXPERIMENT_NAME = "demo-client-juges-gepa"
MIN_DATASET_EXAMPLES = 5

mlflow.set_tracking_uri(MLFLOW_URL)

# Désactive l'autolog OpenAI (cause des bugs downstream avec le streaming SSE)
try:
    mlflow.openai.autolog(disable=True)
except Exception:
    pass
os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"


# ── Patch litellm pour forcer le streaming (compat endpoint SSE-only) ─────────
def _patch_litellm_for_streaming():
    """Force stream=True dans tous les appels litellm (GEPA reflection model).

    Injecte aussi /no_think en système pour désactiver les thinking tokens Qwen3.
    """
    try:
        import litellm
        from litellm import ModelResponse
        from litellm.utils import Choices, Message

        _orig_completion = litellm.completion

        def _patched(*args, **kwargs):
            # Injecte /no_think pour désactiver les thinking tokens Qwen3
            messages = list(kwargs.get("messages", []))
            has_nothink = any(
                m.get("role") == "system" and "/no_think" in m.get("content", "")
                for m in messages
            )
            if not has_nothink:
                messages = [{"role": "system", "content": "/no_think"}] + messages
                kwargs["messages"] = messages

            if kwargs.get("stream"):
                return _orig_completion(*args, **kwargs)
            kwargs["stream"] = True
            content = ""
            finish_reason = "stop"
            try:
                stream = _orig_completion(*args, **kwargs)
                for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            content += delta.content
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
            except Exception as e:
                print(f"   [litellm patch ERROR] {type(e).__name__}: {e}")
                raise
            resp = ModelResponse(
                choices=[Choices(
                    message=Message(role="assistant", content=content),
                    finish_reason=finish_reason,
                )],
                model=kwargs.get("model", ""),
            )
            return resp

        litellm.completion = _patched
        print("[init] litellm patché pour streaming + /no_think Qwen3")
    except Exception as e:
        print(f"[init] litellm patch impossible : {e}")


_patch_litellm_for_streaming()


# ── Client LLM pour les juges ─────────────────────────────────────────────────
def _make_client(timeout: float = 300.0) -> OpenAI:
    return OpenAI(base_url=REMOTE_API_BASE, api_key=API_KEY, timeout=timeout)


def _llm_call(
    messages: list[dict],
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    timeout: float = 300.0,
) -> str:
    """Appel LLM streaming avec retry. Injecte /no_think pour Qwen3."""
    # Injecte /no_think si pas déjà présent
    has_nothink = any(
        m.get("role") == "system" and "/no_think" in m.get("content", "")
        for m in messages
    )
    if not has_nothink:
        messages = [{"role": "system", "content": "/no_think"}] + messages

    last_error: object = None
    for attempt in range(1, max_retries + 1):
        content = ""
        try:
            client = _make_client(timeout=timeout)
            for chunk in client.chat.completions.create(
                model=JUDGE_MODEL,
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
            print(f"   [retry {attempt}/{max_retries}] réponse vide, retry dans {retry_delay}s")
        except Exception as exc:
            last_error = exc
            print(f"   [retry {attempt}/{max_retries}] {type(exc).__name__}: {str(exc)[:200]}")
        if attempt < max_retries:
            _time.sleep(retry_delay)
    raise RuntimeError(f"LLM échoué après {max_retries} tentatives : {last_error}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", text or "")
    return normalized.encode("ascii", "ignore").decode().upper()


def extract_verdict(text: str) -> str:
    normalized = _normalize(text)
    verdict_patterns = [
        (
            "INSUFFISANT",
            [
                r"VERDICT\s*:?\s*[\*#\-\s]*INSUFFISANT",
                r"VERDICT\s*:?\s*[\*#\-\s]*NON SATISFAISANT",
                r"\bNON SATISFAISANT\b",
                r"\bINSUFFISANT\b",
            ],
        ),
        (
            "À AMÉLIORER",
            [
                r"VERDICT\s*:?\s*[\*#\-\s]*A AMELIORER",
                r"\bA AMELIORER\b",
                r"\bAMELIORER\b",
            ],
        ),
        (
            "SATISFAISANT",
            [
                r"VERDICT\s*:?\s*[\*#\-\s]*SATISFAISANT",
                r"\bSATISFAISANT\b",
            ],
        ),
    ]

    for label, patterns in verdict_patterns:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label
    return "INCONNU"


def derive_expected_verdict(judge_verdict: str, human_feedback: str) -> str:
    """Déduit le verdict attendu depuis le verdict du juge + feedback humain."""
    feedback = _normalize(human_feedback).strip()
    if feedback in ("", "OK", "RAS", "ACCORD", "OK.", "RAS.", "OUI"):
        return extract_verdict(judge_verdict)
    if "NON SATISFAISANT" in feedback or "INSATISFAISANT" in feedback or "INSUFFISANT" in feedback:
        return "INSUFFISANT"
    if "A AMELIORER" in feedback or "AMELIORER" in feedback:
        return "À AMÉLIORER"
    if "SATISFAISANT" in feedback:
        return "SATISFAISANT"
    return "À AMÉLIORER"


# ── Chargement de l'historique ────────────────────────────────────────────────
def _history_sort_key(path: Path) -> int:
    m = re.search(r"history_iteration_(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def _read_artifact(mlf_client: MlflowClient, run_id: str, artifact_name: str) -> str:
    try:
        artifact_path = mlf_client.download_artifacts(run_id, artifact_name)
        return Path(artifact_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_trace_info(trace: dict) -> str:
    if trace.get("trace_info"):
        return str(trace["trace_info"])

    n_visited = int(_safe_float(trace.get("n_visited", 0)))
    n_notes = int(_safe_float(trace.get("n_notes", 0)))
    elapsed = _safe_float(trace.get("elapsed", 0.0))
    return "\n".join(
        [
            f"Noeuds visités ({n_visited}) :",
            str(trace.get("visited_nodes") or "(aucun noeud visité)"),
            "",
            f"Chemin de navigation : {trace.get('nav_path') or '--'}",
            "",
            f"Notes de l'agent ({n_notes}) :",
            str(trace.get("notes") or "(aucune note enregistrée)"),
            "",
            f"Durée : {elapsed:.1f}s",
        ]
    )


def _hydrate_trace_record(trace: dict, mlf_client: MlflowClient) -> dict:
    hydrated = dict(trace)
    run_id = str(hydrated.get("run_id", "") or "")
    if run_id:
        try:
            run = mlf_client.get_run(run_id)
            metrics = run.data.metrics
            hydrated.setdefault("n_visited", metrics.get("n_visited_nodes", 0.0))
            hydrated.setdefault("n_notes", metrics.get("n_notes", 0.0))
            hydrated.setdefault("elapsed", metrics.get("elapsed_seconds", 0.0))
        except Exception:
            pass

        if not hydrated.get("visited_nodes"):
            hydrated["visited_nodes"] = _read_artifact(mlf_client, run_id, "visited_nodes.txt").strip()
        if not hydrated.get("notes"):
            hydrated["notes"] = _read_artifact(mlf_client, run_id, "notes.txt").strip()
        if not hydrated.get("nav_path"):
            hydrated["nav_path"] = _read_artifact(mlf_client, run_id, "navigation_path.txt").strip()

    hydrated["trace_info"] = _build_trace_info(hydrated)
    return hydrated


def load_history() -> list[dict]:
    """Charge le dernier history_iteration_N.json disponible."""
    files = sorted(ARTIFACTS_DIR.glob("history_iteration_*.json"), key=_history_sort_key)
    if not files:
        raise FileNotFoundError(
            f"Aucun history_iteration_*.json dans {ARTIFACTS_DIR}.\n"
            "Lance d'abord : python -m demo_client.3_juges_iteratif"
        )
    latest = files[-1]
    print(f"  Historique chargé : {latest.name}")
    return json.loads(latest.read_text(encoding="utf-8"))


def build_dataset(history: list[dict], judge_name: str) -> list[dict]:
    """Construit le dataset {inputs, expectations} pour un juge donné."""
    dataset_by_run: dict[str, dict] = {}
    mlf_client = MlflowClient()

    for trace in history:
        judges = trace.get("judges", {})
        if judge_name not in judges:
            continue
        judge_data = judges[judge_name]
        expected = derive_expected_verdict(
            str(judge_data.get("verdict", "")),
            str(judge_data.get("human_feedback", "")),
        )
        if expected not in VERDICTS:
            continue

        hydrated = _hydrate_trace_record(trace, mlf_client)
        question = str(hydrated.get("question", "")).strip()
        answer = str(hydrated.get("answer", "")).strip()
        trace_info = str(hydrated.get("trace_info", "")).strip()
        if not question or not answer or not trace_info:
            continue

        run_key = str(hydrated.get("run_id") or f"{question[:120]}::{judge_name}")
        dataset_by_run[run_key] = {
            "inputs": {
                "question": question,
                "answer": answer,
                "trace_info": trace_info,
            },
            "expectations": {
                "verdict": expected,
                "human_feedback": str(judge_data.get("human_feedback", "")),
            },
        }

    dataset = list(dataset_by_run.values())
    print(f"  [dataset] {judge_name}: {len(dataset)} exemple(s) utilisable(s)")
    return dataset


# ── Predict function : exécute le juge avec le dernier prompt du Registry ─────
def make_predict_fn(prompt_name: str):
    """Crée une closure qui exécute le juge avec la dernière version du prompt."""
    _mlf = MlflowClient()

    def predict_fn(question: str, answer: str, trace_info: str) -> str:
        versions = _mlf.search_prompt_versions(name=prompt_name)
        if not versions:
            raise RuntimeError(f"Aucune version pour le prompt '{prompt_name}'")
        latest_v = max(int(v.version) for v in versions)
        prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/{latest_v}")
        filled = prompt.format(inputs=question, outputs=answer, trace=trace_info)
        return _llm_call(
            messages=[{"role": "user", "content": filled}],
            temperature=0.0,
        )

    return predict_fn


# ── Scorers ───────────────────────────────────────────────────────────────────
@scorer
def verdict_match(outputs: str, expectations: dict) -> float:
    """1.0 si le verdict prédit correspond au verdict attendu."""
    predicted = extract_verdict(outputs)
    expected = expectations.get("verdict", "INCONNU")
    return 1.0 if predicted == expected else 0.0


@scorer
def has_required_structure(outputs: str) -> float:
    """Vérifie que la réponse contient ANALYSE + VERDICT + RECOMMANDATION."""
    n = _normalize(outputs)
    return float("ANALYSE" in n and "VERDICT" in n and "RECOMMANDATION" in n)


_RUBRIC_PROMPT = """\
Tu évalues l'alignement sémantique entre la recommandation produite par un juge IA
et la critique formulée par un évaluateur humain expert sur la même réponse.

CRITIQUE HUMAINE ATTENDUE :
{human_feedback}

RÉPONSE COMPLÈTE DU JUGE IA (analyse + verdict + recommandation) :
{judge_output}

RUBRIQUE (0.0 à 1.0) :
- 1.0 : le juge identifie les MÊMES problèmes que l'humain et propose des actions concrètes alignées
- 0.7 : le juge couvre l'idée principale mais manque des nuances ou propose des actions partielles
- 0.4 : le juge effleure le sujet mais passe à côté du point clé de l'humain
- 0.1 : recommandation générique ou hors-sujet
- 0.0 : aucun alignement ou recommandation absente

Si la critique humaine est vide / "ok", évalue si la recommandation est pertinente et non-générique :
  1.0 si pertinente, 0.5 si générique, 0.0 si hors-sujet.

Réponds UNIQUEMENT par un nombre décimal entre 0.0 et 1.0, sans texte."""


@scorer
def recommendation_alignment(outputs: str, expectations: dict) -> float:
    """LLM-as-judge : aligne la recommandation du juge avec le feedback humain."""
    human_fb = expectations.get("human_feedback", "").strip()
    try:
        raw = _llm_call(
            messages=[{
                "role": "user",
                "content": _RUBRIC_PROMPT.format(
                    human_feedback=human_fb or "(aucun feedback — évaluer la pertinence intrinsèque)",
                    judge_output=outputs,
                ),
            }],
            temperature=0.0,
            timeout=120.0,
        )
        match = re.search(r"[0-9]*\.?[0-9]+", raw)
        if match:
            return max(0.0, min(1.0, float(match.group())))
    except Exception as e:
        print(f"   [scorer recommendation_alignment ERROR] {e}")
    return 0.0


# ── Update juge dans MLflow UI ────────────────────────────────────────────────
def _update_judge_ui(judge_name: str, new_instructions: str) -> bool:
    """Met à jour le juge dans MLflow UI (section Judges) après optimisation GEPA."""
    try:
        from mlflow.genai.judges import make_judge
        from mlflow.genai.scorers import list_scorers

        exp = mlflow.get_experiment_by_name("demo-client-agent")
        exp_id = exp.experiment_id if exp else None
        if not exp_id:
            print(f"  [UI] Expérience 'demo-client-agent' introuvable")
            return False

        # Récupère le modèle existant du juge
        model = f"openai:/{JUDGE_MODEL}"
        try:
            for s in list_scorers(experiment_id=exp_id):
                if getattr(s, "name", None) == judge_name:
                    model = getattr(s, "model", model) or model
                    break
        except Exception:
            pass

        new_judge = make_judge(name=judge_name, instructions=new_instructions, model=model)
        new_judge.register(experiment_id=exp_id)
        print(f"  [UI] Juge '{judge_name}' mis à jour ✓")
        return True
    except Exception as exc:
        print(f"  [UI] Échec mise à jour '{judge_name}' : {exc}")
        return False


# ── Optimisation d'un juge ────────────────────────────────────────────────────
def optimize_judge(judge_name: str, dataset: list[dict]) -> str | None:
    """Lance GEPA sur le prompt du juge. Retourne l'URI de la nouvelle version ou None."""
    prompt_name = f"{JUDGE_PREFIX}-{judge_name}"

    # Charge la dernière version du prompt
    try:
        mlf_client = MlflowClient()
        versions = mlf_client.search_prompt_versions(name=prompt_name)
        if not versions:
            print(f"  [SKIP] Prompt '{prompt_name}' : aucune version dans le Registry")
            return None
        latest_version = max(int(v.version) for v in versions)
        latest_uri = f"prompts:/{prompt_name}/{latest_version}"
        mlflow.genai.load_prompt(latest_uri)
        print(f"  >> Prompt '{prompt_name}' v{latest_version} chargé")
    except Exception as e:
        print(f"  [SKIP] Prompt '{prompt_name}' introuvable : {e}")
        return None

    print(f"\n{'='*70}")
    print(f"  Optimisation GEPA : {judge_name}")
    print(f"{'='*70}")
    print(f"  Exemples       : {len(dataset)}")
    print(f"  Verdicts attend.: {[d['expectations']['verdict'] for d in dataset]}")
    print(f"  Reflection model: {REFLECTION_MODEL}")

    optimizer = GepaPromptOptimizer(
        reflection_model=REFLECTION_MODEL,
        max_metric_calls=60,
        display_progress_bar=True,
    )

    try:
        result = optimize_prompts(
            predict_fn=make_predict_fn(prompt_name),
            train_data=dataset,
            prompt_uris=[latest_uri],
            optimizer=optimizer,
            scorers=[verdict_match, has_required_structure, recommendation_alignment],
        )

        # Extraction du template optimisé
        optimized_template = None
        if hasattr(result, "optimized_prompts") and result.optimized_prompts:
            opt = result.optimized_prompts[0]
            optimized_template = (
                getattr(opt, "template", None)
                or getattr(opt, "text", None)
                or getattr(opt, "content", None)
                or (opt.prompt.template if hasattr(opt, "prompt") and hasattr(opt.prompt, "template") else None)
            )

        if not optimized_template:
            print(f"  [WARN] Template optimisé introuvable pour {judge_name}")
            return None

        # Réparation des variables Jinja supprimées par GEPA
        REQUIRED_VARS = ["{{ inputs }}", "{{ outputs }}", "{{ trace }}"]
        missing_vars = [v for v in REQUIRED_VARS if v not in optimized_template]
        if missing_vars:
            print(f"  [REPAIR] Variables Jinja supprimées par GEPA : {missing_vars}")
            data_block = (
                "\n\n============================================================\n"
                "DONNÉES À ÉVALUER\n"
                "============================================================\n"
                "Question posée : {{ inputs }}\n"
                "Réponse de l'agent : {{ outputs }}\n"
                "Trace complète du raisonnement et des outils appelés : {{ trace }}\n"
            )
            optimized_template = optimized_template.rstrip() + data_block
            print(f"  [REPAIR] Bloc DONNÉES réinjecté en fin de prompt")

        # Vérifie si le prompt a vraiment changé
        baseline_template = mlflow.genai.load_prompt(latest_uri).template
        if optimized_template.strip() == baseline_template.strip():
            print(f"  [INFO] Prompt identique à la baseline — GEPA n'a rien modifié")
            if hasattr(result, "initial_eval_score_per_scorer"):
                print(f"         Initial scores : {result.initial_eval_score_per_scorer}")
            if hasattr(result, "final_eval_score_per_scorer"):
                print(f"         Final scores   : {result.final_eval_score_per_scorer}")
            return None

        # Diff visuel
        diff_chars = sum(1 for a, b in zip(baseline_template, optimized_template) if a != b)
        size_delta = len(optimized_template) - len(baseline_template)
        print(f"  >> DIFF : {diff_chars} caractères modifiés, delta taille = {size_delta:+d} chars")

        score = getattr(result, "final_eval_score", "?")
        repaired_flag = " + Jinja vars restaurées" if missing_vars else ""

        # Enregistre la nouvelle version dans le Prompt Registry
        new_version = mlflow.genai.register_prompt(
            name=prompt_name,
            template=optimized_template,
            commit_message=f"GEPA optimization (score={score}){repaired_flag}",
            tags={
                "optimizer": "GEPA",
                "source_version": str(latest_version),
                "jinja_repaired": "true" if missing_vars else "false",
            },
        )
        new_uri = f"prompts:/{prompt_name}/{new_version.version}"
        print(f"  ✓ Nouvelle version : {new_uri}")

        # Sauvegarde du diff pour inspection
        diff_path = ARTIFACTS_DIR / f"diff_{judge_name}_v{latest_version}_to_gepa.txt"
        diff_path.write_text(
            f"=== BASELINE (v{latest_version}) ===\n{baseline_template}\n\n"
            f"=== GEPA OPTIMIZED (v{new_version.version}) ===\n{optimized_template}\n",
            encoding="utf-8",
        )
        print(f"     Diff sauvegardé : {diff_path.name}")

        # Met aussi à jour le juge dans MLflow UI
        _update_judge_ui(judge_name, optimized_template)

        return new_uri

    except Exception as exc:
        import traceback
        print(f"  [ERREUR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 70)
    print("  GEPA — Optimisation des prompts de juges")
    print("=" * 70)
    print(f"  MLflow        : {MLFLOW_URL}")
    print(f"  Endpoint LLM  : {REMOTE_API_BASE}")
    print(f"  Modèle juge   : {JUDGE_MODEL}")
    print(f"  Reflection    : {REFLECTION_MODEL}")
    print(f"  Juges         : {JUDGES}")
    print()

    history = load_history()
    print(f"  {len(history)} trace(s) chargée(s)\n")

    with mlflow.start_run(run_name="gepa_judges_optimization"):
        mlflow.log_param("nb_traces", len(history))
        mlflow.log_param("optimizer", "GepaPromptOptimizer")
        mlflow.log_param("reflection_model", REFLECTION_MODEL)
        mlflow.log_param("judge_model", JUDGE_MODEL)
        mlflow.log_param("judges", ", ".join(JUDGES))

        results = {}
        for judge_name in JUDGES:
            dataset = build_dataset(history, judge_name)
            verdict_counts = {label: sum(1 for item in dataset if item["expectations"]["verdict"] == label) for label in VERDICTS}
            mlflow.log_metric(f"{judge_name}_dataset_size", len(dataset))
            for label, count in verdict_counts.items():
                mlflow.log_metric(f"{judge_name}_{_normalize(label).lower().replace(' ', '_')}", count)

            if len(dataset) < MIN_DATASET_EXAMPLES:
                print(
                    f"\n[SKIP] {judge_name} : seulement {len(dataset)} exemple(s) utilisable(s) "
                    f"(minimum {MIN_DATASET_EXAMPLES})"
                )
                results[judge_name] = None
                continue

            new_uri = optimize_judge(judge_name, dataset)
            results[judge_name] = new_uri
            if new_uri:
                mlflow.log_param(f"{judge_name}_new_uri", new_uri)

    print("\n" + "=" * 70)
    print("  RÉSUMÉ GEPA JUGES")
    print("=" * 70)
    for j, uri in results.items():
        status = uri if uri else "inchangé / échec"
        print(f"  {j}: {status}")
    print(f"\nVois les nouvelles versions sur {MLFLOW_URL}/#/prompts")
