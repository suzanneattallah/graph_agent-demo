"""
Generates score evolution charts from judge iteration history files.
Reads history_iteration_N.json files from the artifact dir and creates:
1. Line chart: % SATISFAISANT per judge across iterations
2. Bar chart: verdict distribution (SATISFAISANT / À AMÉLIORER / INSUFFISANT) per iteration
3. Radar chart: latest iteration scores per judge
Saves charts as PNG and logs to MLflow.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .config import ARTIFACTS_DIR, JUDGES, JUDGE_EXPERIMENT, MLFLOW_URL
except ImportError:
    from demo_client.config import ARTIFACTS_DIR, JUDGES, JUDGE_EXPERIMENT, MLFLOW_URL

OUTPUT_FILE = ARTIFACTS_DIR / "scores_evolution.png"
SCORE_BY_VERDICT = {"SATISFAISANT": 1.0, "À AMÉLIORER": 0.5, "INSUFFISANT": 0.0, "INCONNU": 0.0}
VERDICT_ORDER = ["SATISFAISANT", "À AMÉLIORER", "INSUFFISANT"]
VERDICT_COLORS = {"SATISFAISANT": "#2ca02c", "À AMÉLIORER": "#ffbf00", "INSUFFISANT": "#d62728"}


def extract_verdict(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode().upper()
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*SATISFAISANT", normalized):
        return "SATISFAISANT"
    if re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", normalized):
        match = re.search(r"VERDICT\s*:?\s*[\*#]*\s*(A AMELIORER|INSUFFISANT)", normalized)
        return "INSUFFISANT" if "INSUFFISANT" in (match.group(1) if match else "") else "À AMÉLIORER"
    if "SATISFAISANT" in normalized:
        return "SATISFAISANT"
    if "A AMELIORER" in normalized:
        return "À AMÉLIORER"
    if "INSUFFISANT" in normalized:
        return "INSUFFISANT"
    return "INCONNU"


def _history_sort_key(path: Path) -> int:
    match = re.search(r"history_iteration_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def _load_history_files() -> list[Path]:
    return sorted(ARTIFACTS_DIR.glob("history_iteration_*.json"), key=_history_sort_key)


def _aggregate_iteration(path: Path) -> dict[str, object]:
    history = json.loads(path.read_text(encoding="utf-8"))
    verdict_counts = {verdict: 0 for verdict in VERDICT_ORDER}
    judge_totals = {judge: 0 for judge in JUDGES}
    judge_satisfying = {judge: 0 for judge in JUDGES}
    judge_scores: dict[str, list[float]] = {judge: [] for judge in JUDGES}
    n_visited_vals: list[int] = []
    n_layers_vals: list[int] = []

    for entry in history:
        for judge in JUDGES:
            judge_data = entry.get("judges", {}).get(judge, {})
            raw_verdict = judge_data.get("verdict", "")

            # Utilise le score numérique stocké si disponible, sinon extrait
            score = judge_data.get("score")
            if score is None:
                score = SCORE_BY_VERDICT.get(extract_verdict(raw_verdict), 0.0)

            verdict = judge_data.get("verdict_label") or extract_verdict(raw_verdict)
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
            judge_totals[judge] += 1
            if verdict == "SATISFAISANT":
                judge_satisfying[judge] += 1
            judge_scores[judge].append(float(score))

            # Extrait n_visited / n_layers depuis JugeExploration
            if judge == "JugeExploration":
                nv = re.search(r"N_VISITED\s*[=:]\s*(\d+)", raw_verdict, re.IGNORECASE)
                nl = re.search(r"N_LAYERS\s*[=:]\s*(\d+)", raw_verdict, re.IGNORECASE)
                if nv:
                    n_visited_vals.append(int(nv.group(1)))
                if nl:
                    n_layers_vals.append(int(nl.group(1)))

    judge_pct = {
        judge: (judge_satisfying[judge] / judge_totals[judge] * 100.0) if judge_totals[judge] else 0.0
        for judge in JUDGES
    }
    global_total = sum(verdict_counts.values())
    global_pct = (verdict_counts["SATISFAISANT"] / global_total * 100.0) if global_total else 0.0
    latest_scores = {
        judge: (sum(judge_scores[judge]) / len(judge_scores[judge]) * 100.0) if judge_scores[judge] else 0.0
        for judge in JUDGES
    }

    return {
        "iteration":    _history_sort_key(path),
        "judge_pct":    judge_pct,
        "global_pct":   global_pct,
        "verdict_counts": verdict_counts,
        "latest_scores":  latest_scores,
        "avg_n_visited":  sum(n_visited_vals) / len(n_visited_vals) if n_visited_vals else None,
        "avg_n_layers":   sum(n_layers_vals)  / len(n_layers_vals)  if n_layers_vals  else None,
    }


def _build_empty_figure(message: str) -> Path:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 6))
    figure.text(0.5, 0.5, message, ha="center", va="center", fontsize=16)
    figure.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT_FILE


def generate_charts() -> Path:
    history_files = _load_history_files()
    if not history_files:
        print(f"Aucun fichier history_iteration_*.json dans {ARTIFACTS_DIR}")
        output = _build_empty_figure("No judge iteration history found.")
    else:
        import matplotlib.pyplot as plt

        aggregated = [_aggregate_iteration(path) for path in history_files]
        iterations = [item["iteration"] for item in aggregated]

        figure = plt.figure(figsize=(20, 10))
        ax_line  = figure.add_subplot(2, 3, 1)
        ax_bar   = figure.add_subplot(2, 3, 2)
        ax_radar = figure.add_subplot(2, 3, 3, polar=True)
        ax_expl  = figure.add_subplot(2, 3, 4)
        ax_score = figure.add_subplot(2, 3, 5)

        # ── Graphe 1 : % SATISFAISANT par juge (évolution) ────────────────────
        for judge in JUDGES:
            ax_line.plot(iterations, [item["judge_pct"][judge] for item in aggregated], marker="o", label=judge)
        ax_line.plot(
            iterations,
            [item["global_pct"] for item in aggregated],
            marker="o", linewidth=3, color="black", label="global",
        )
        ax_line.set_title("% SATISFAISANT par juge")
        ax_line.set_xlabel("Itération")
        ax_line.set_ylabel("% SATISFAISANT")
        ax_line.set_ylim(0, 100)
        ax_line.set_xticks(iterations)
        ax_line.grid(True, linestyle=":", alpha=0.5)
        ax_line.legend(fontsize=8)

        # ── Graphe 2 : distribution des verdicts (empilé) ─────────────────────
        bottoms = [0] * len(iterations)
        for verdict in VERDICT_ORDER:
            values = [item["verdict_counts"][verdict] for item in aggregated]
            ax_bar.bar(iterations, values, bottom=bottoms, color=VERDICT_COLORS[verdict], label=verdict)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax_bar.set_title("Distribution des verdicts")
        ax_bar.set_xlabel("Itération")
        ax_bar.set_ylabel("Nombre de verdicts")
        ax_bar.set_xticks(iterations)
        ax_bar.legend(fontsize=8)

        # ── Graphe 3 : radar (scores moyens dernière itération) ───────────────
        latest = aggregated[-1]
        labels = JUDGES[:]
        angles = [2 * 3.141592653589793 * i / len(labels) for i in range(len(labels))]
        angles += angles[:1]
        values = [latest["latest_scores"][j] for j in labels] + [latest["latest_scores"][labels[0]]]
        ax_radar.plot(angles, values, color="#1f77b4", linewidth=2)
        ax_radar.fill(angles, values, color="#1f77b4", alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels, fontsize=7)
        ax_radar.set_yticks([25, 50, 75, 100])
        ax_radar.set_yticklabels(["25", "50", "75", "100"], fontsize=7)
        ax_radar.set_ylim(0, 100)
        ax_radar.set_title(f"Scores moyens — itération {latest['iteration']}", pad=15)

        # ── Graphe 4 : n_visited / n_layers (métriques d'exploration objectives)
        nv_vals = [item["avg_n_visited"] for item in aggregated if item.get("avg_n_visited") is not None]
        nl_vals = [item["avg_n_layers"]  for item in aggregated if item.get("avg_n_layers")  is not None]
        iter_nv  = [item["iteration"]    for item in aggregated if item.get("avg_n_visited") is not None]
        iter_nl  = [item["iteration"]    for item in aggregated if item.get("avg_n_layers")  is not None]
        if nv_vals:
            ax_expl.plot(iter_nv, nv_vals, marker="s", color="#ff7f0e", label="Nœuds visités (moy.)")
        if nl_vals:
            ax_expl.plot(iter_nl, nl_vals, marker="^", color="#2ca02c", label="Layers couverts (moy.)")
        ax_expl.axhline(y=5, color="#ff7f0e", linestyle="--", alpha=0.4, label="Seuil n_visited=5")
        ax_expl.axhline(y=2, color="#2ca02c", linestyle="--", alpha=0.4, label="Seuil n_layers=2")
        ax_expl.set_title("Métriques d'exploration (JugeExploration)")
        ax_expl.set_xlabel("Itération")
        ax_expl.set_ylabel("Valeur moyenne")
        ax_expl.set_xticks(iterations)
        ax_expl.grid(True, linestyle=":", alpha=0.5)
        ax_expl.legend(fontsize=8)

        # ── Graphe 5 : score numérique moyen par juge (0.0→1.0) ───────────────
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for idx, judge in enumerate(JUDGES):
            scores = [item["latest_scores"][judge] / 100.0 for item in aggregated]
            ax_score.plot(iterations, scores, marker="o", color=colors[idx % len(colors)], label=judge)
        ax_score.set_title("Score numérique moyen par juge (0→1)")
        ax_score.set_xlabel("Itération")
        ax_score.set_ylabel("Score moyen")
        ax_score.set_ylim(0, 1.05)
        ax_score.set_xticks(iterations)
        ax_score.axhline(y=1.0, color="green",  linestyle="--", alpha=0.3, label="SATISFAISANT")
        ax_score.axhline(y=0.5, color="orange", linestyle="--", alpha=0.3, label="À AMÉLIORER")
        ax_score.grid(True, linestyle=":", alpha=0.5)
        ax_score.legend(fontsize=8)

        figure.suptitle("GraphAgent — Évolution des métriques d'évaluation", fontsize=14, fontweight="bold")
        figure.tight_layout()
        figure.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        plt.close(figure)
        output = OUTPUT_FILE
        print(f"Graphique généré : {output}")

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(JUDGE_EXPERIMENT)
    with mlflow.start_run(run_name="scores_evolution_charts"):
        mlflow.log_param("history_files", len(history_files))
        mlflow.log_artifact(str(output))

    return output


def main() -> Path:
    return generate_charts()


if __name__ == "__main__":
    main()
