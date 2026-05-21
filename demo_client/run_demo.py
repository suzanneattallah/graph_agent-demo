from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(filename: str, module_name: str):
    module_path = PACKAGE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _separator(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _pause(auto: bool) -> None:
    if auto:
        return
    try:
        input("\nAppuyez sur Entrée pour continuer...")
    except EOFError:
        print("\n[warn] Entrée non interactive détectée, poursuite automatique.")


def main(iterations: int = 3, auto: bool = False, skip_traces: bool = False, skip_judges: bool = False) -> None:
    graph_stats = _load_module("1_graph_stats.py", "demo_client_step_graph_stats")
    generer_traces = _load_module("2_generer_traces.py", "demo_client_step_generer_traces")
    juges_iteratif = _load_module("3_juges_iteratif.py", "demo_client_step_juges_iteratif")
    optimize_agent = _load_module("4_optimize_agent.py", "demo_client_step_optimize_agent")
    mlflow_charts = _load_module("5_mlflow_charts.py", "demo_client_step_mlflow_charts")

    _separator("STEP 1: Graph stats")
    graph_stats.show_graph_stats()
    _pause(auto)

    if not skip_traces:
        _separator("STEP 2: Generate traces")
        generer_traces.run_all()
        _pause(auto)
    else:
        print("\nSTEP 2 skipped (--skip-traces)")

    if not skip_judges:
        _separator("STEP 3: Judge iterations loop")
        juges_iteratif.main(limit=5, iterations=iterations)
        _pause(auto)
    else:
        print("\nSTEP 3 skipped (--skip-judges)")

    _separator("STEP 4: Optimize agent")
    optimize_agent.main(do_retest=False, max_iter=iterations)
    _pause(auto)

    _separator("STEP 5: Generate charts")
    mlflow_charts.generate_charts()
    print("\nDemo pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete demo_client pipeline")
    parser.add_argument("--iterations", type=int, default=3, help="Nombre d'itérations pour les juges")
    parser.add_argument("--auto", action="store_true", help="N'attend pas entre les étapes")
    parser.add_argument("--skip-traces", action="store_true", help="Ignore la génération de traces")
    parser.add_argument("--skip-judges", action="store_true", help="Ignore la boucle des juges")
    args = parser.parse_args()
    main(
        iterations=args.iterations,
        auto=args.auto,
        skip_traces=args.skip_traces,
        skip_judges=args.skip_judges,
    )
