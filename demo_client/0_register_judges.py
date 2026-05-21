"""
Enregistre les 4 juges du demo GraphAgent dans MLflow.

Ce script doit être lancé UNE SEULE FOIS (ou après réinitialisation de la DB).
Il crée :
  1. Les prompts initiaux dans le Prompt Registry (section Prompts de MLflow UI)
  2. Les 4 juges dans l'experiment demo-client-agent (section Judges de MLflow UI)

Usage :
  python -m demo_client.0_register_judges
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from mlflow.genai.judges import make_judge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import AGENT_EXPERIMENT, JUDGE_PREFIX, MLFLOW_URL, REMOTE_API_BASE
except ImportError:
    from demo_client.config import AGENT_EXPERIMENT, JUDGE_PREFIX, MLFLOW_URL, REMOTE_API_BASE

from graph_agent.juges.register_juges import FALLBACK_PROMPTS

mlflow.set_tracking_uri(MLFLOW_URL)

JUDGE_MODEL = f"openai/{REMOTE_API_BASE.rstrip('/v1').rstrip('/')}"


def register_initial_prompts() -> None:
    """Enregistre la version initiale de chaque prompt juge dans le Prompt Registry."""
    print("\n── Prompt Registry ──────────────────────────────────────────────────")
    for judge_name, template in FALLBACK_PROMPTS.items():
        prompt_name = f"{JUDGE_PREFIX}-{judge_name}"
        try:
            v = mlflow.genai.register_prompt(
                name=prompt_name,
                template=template,
                commit_message="v1 : prompt initial (bootstrap)",
                tags={"judge": judge_name, "source": "FALLBACK_PROMPTS", "project": "demo-client"},
            )
            print(f"  ✅ {prompt_name} → v{v.version}")
        except Exception as exc:
            print(f"  ⚠️  {prompt_name} : {type(exc).__name__}: {exc}")


def register_judges_ui() -> None:
    """Crée les 4 juges dans l'UI MLflow (section Judges de l'experiment)."""
    print("\n── MLflow Judges UI ─────────────────────────────────────────────────")
    exp = mlflow.get_experiment_by_name(AGENT_EXPERIMENT)
    if exp is None:
        print(f"  ❌ Experiment '{AGENT_EXPERIMENT}' introuvable. Lance d'abord 2_generer_traces.")
        return
    exp_id = exp.experiment_id

    for judge_name, instructions in FALLBACK_PROMPTS.items():
        try:
            judge = make_judge(
                name=judge_name,
                instructions=instructions,
                model="openai:/gpt-4o-mini",  # modèle indicatif pour l'UI (exécution reste manuelle)
            )
            judge.register(experiment_id=exp_id)
            print(f"  ✅ Juge '{judge_name}' enregistré dans l'experiment '{AGENT_EXPERIMENT}'")
        except Exception as exc:
            print(f"  ⚠️  Juge '{judge_name}' : {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"MLflow : {MLFLOW_URL}")
    print(f"Experiment : {AGENT_EXPERIMENT}")
    print(f"Prefix prompts : {JUDGE_PREFIX}")

    register_initial_prompts()
    register_judges_ui()

    print(f"\n✅ Terminé !")
    print(f"   Prompts : {MLFLOW_URL}/#/prompts")
    print(f"   Judges  : {MLFLOW_URL}/#/experiments/{mlflow.get_experiment_by_name(AGENT_EXPERIMENT).experiment_id}/judges")
