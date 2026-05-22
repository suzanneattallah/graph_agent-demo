from pathlib import Path

MLFLOW_URL = "http://localhost:5000"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "agent_recherche_neo4j"

AGENT_EXPERIMENT = "demo-client-agent"
JUDGE_EXPERIMENT = "demo-client-juges"
AGENT_PROMPT_NAME = "demo-client-SystemPrompt"
JUDGE_PREFIX = "demo-client"
ARTIFACTS_DIR = Path(__file__).parent / "artifact"

REMOTE_API_BASE = "http://CHLASLITASSAPR1.lan.la.sqli.com:8080/v1"
AGENT_MODEL = "qwen3.6-35b-a3b"
JUDGE_MODEL = "qwen3.6-27b"
REFINER_MODEL = "qwen3.6-27b"
API_KEY = "none"

JUDGES = ["JugeExploration", "JugePrecisionTechnique", "JugeRaisonnement", "JugeAmeliorations"]
CONVERGENCE_THRESHOLD = 1.0
MAX_ITERATIONS = 5

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
