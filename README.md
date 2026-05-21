# Graph Agent Demo

Agent DSPy ReAct qui navigue un graphe Neo4j de codebase Java pour répondre à des questions d'analyse de code, avec tracking MLflow et visualisation Streamlit en temps réel.

## Stack
- **LLM** : Ollama local (`qwen2.5-coder:7b`) ou endpoint distant OpenAI-compatible
- **Graphe** : Neo4j 5 (via Podman/Docker)
- **Tracking** : MLflow 3
- **Visualisation** : Streamlit + streamlit-agraph
- **Agent** : DSPy ReAct

## Setup rapide

### 1. Prérequis
```bash
# Python 3.11+
pip install -r requirements.txt

# Podman ou Docker installé
# Ollama installé (optionnel si endpoint distant disponible)
```

### 2. Démarrer Neo4j
```bash
# Podman
podman run -d --name neo4j-graph \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/agent_recherche_neo4j \
  neo4j:5-community

# Docker
docker run -d --name neo4j-graph \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/agent_recherche_neo4j \
  neo4j:5-community
```

### 3. Importer le graphe mall
```bash
python import_to_neo4j.py
# → 14 996 nœuds, 28 129 arêtes importés
```

### 4. Démarrer MLflow
```bash
mlflow server --host 0.0.0.0 --port 5000
# ou via Podman (voir graph_agent/mlflow/)
```

### 5. Lancer l'agent (CLI)
```bash
# Avec Ollama local
python -m graph_agent.run "What does OmsOrderController do?"

# Avec endpoint distant
python -m graph_agent.run "What does OmsOrderController do?" \
  --api-base http://YOUR_ENDPOINT:8080/v1
```

### 6. Visualiseur Streamlit (temps réel)
```bash
streamlit run graph_agent/visualizer.py
# → http://localhost:8501
```

### 7. Pipeline démo client
```bash
# Étape 1 — Stats du graphe
python -m demo_client.1_graph_stats

# Étape 2 — Générer 7 traces MLflow
python -m demo_client.2_generer_traces

# Étape 3 — Boucle juges itérative (interactif)
python -m demo_client.3_juges_iteratif --iterations 2 --limit 3

# Étape 4 — Optimiser le prompt agent
python -m demo_client.4_optimize_agent

# Étape 5 — Charts MLflow
python -m demo_client.5_mlflow_charts
```

## Configuration

Éditer `demo_client/config.py` pour changer :
- `REMOTE_API_BASE` : endpoint LLM distant
- `AGENT_MODEL` / `JUDGE_MODEL` : modèles à utiliser
- `NEO4J_URI` / `NEO4J_PASSWORD` : connexion Neo4j
- `MLFLOW_URL` : URL MLflow

## Architecture

```
graph_agent/
├── agent.py        ← GraphAgent (DSPy ReAct + MLflow)
├── tools.py        ← 13 outils Cypher Neo4j
├── run.py          ← CLI entry point
├── visualizer.py   ← Streamlit temps réel (lit nav_state.json)
└── juges/          ← Pipeline d'évaluation itératif

demo_client/
├── config.py              ← Config centralisée
├── 1_graph_stats.py       ← Stats Neo4j
├── 2_generer_traces.py    ← Génération de traces
├── 3_juges_iteratif.py    ← Boucle juges + feedback humain
├── 4_optimize_agent.py    ← Optimisation prompt agent
└── 5_mlflow_charts.py     ← Courbes d'évolution

import_to_neo4j.py   ← Import mall-ast.json → Neo4j
mall-ast.json        ← AST du projet macrozheng/mall (14 996 nœuds)
```

## Notes Mac/Linux

Les chemins Windows (`C:\Projet\...`) dans les scripts sont relatifs au projet — pas de modification nécessaire.
MLflow port : **5000** | Neo4j bolt : **7687** | Streamlit : **8501**
