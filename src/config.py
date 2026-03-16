"""Configuration for Full-TS Cognitive Architecture."""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
MEMORY_DIR = PROJECT_ROOT / "memory"
EVAL_DIR = PROJECT_ROOT / "eval"
VIZ_DIR = PROJECT_ROOT / "viz"

# Ensure directories exist
for d in [DATA_DIR, GRAPHS_DIR, MODELS_DIR, MEMORY_DIR, EVAL_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sub in ["inputs", "raw"]:
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
for sub in ["raw", "structured", "hypothesis", "meta"]:
    (MEMORY_DIR / sub).mkdir(parents=True, exist_ok=True)

# Graph DB (Memgraph/Neo4j compatible)
GRAPH_URI = os.getenv("GRAPH_URI", "bolt://localhost:7687")
GRAPH_USER = os.getenv("GRAPH_USER", "")
GRAPH_PASSWORD = os.getenv("GRAPH_PASSWORD", "")

# Redis (Celery broker)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"

# Fallback to NetworkX when graph DB unavailable
USE_NETWORKX_FALLBACK = os.getenv("USE_NETWORKX_FALLBACK", "true").lower() == "true"
NETWORKX_MAX_NODES = int(os.getenv("NETWORKX_MAX_NODES", "100000"))

# Large-graph scaling: auto-switch to Memgraph above threshold
ENABLE_LARGE_GRAPH = os.getenv("ENABLE_LARGE_GRAPH", "true").lower() == "true"
MIN_NODES_FOR_MEMGRAPH = int(os.getenv("MIN_NODES_FOR_MEMGRAPH", "50000"))

# Parallel thinking (Ollama instances on different ports)
def _cpu_cores() -> int:
    """Return CPU core count. Safe when psutil is not installed."""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except ImportError:
        return 1
    except Exception:
        return 1
_PARALLEL_RAW = int(os.getenv("PARALLEL_THREADS", "0"))
PARALLEL_THREADS = max(1, _PARALLEL_RAW) if _PARALLEL_RAW > 0 else max(1, _cpu_cores() // 2)

# Coherence at scale: prune low-degree nodes below this
PRUNE_DEGREE_THRESHOLD = int(os.getenv("PRUNE_DEGREE_THRESHOLD", "1"))
RESOURCE_PRESSURE_THRESHOLD = float(os.getenv("RESOURCE_PRESSURE_THRESHOLD", "90.0"))

# Ethical filters
HARMFUL_PATTERNS = [
    "harm", "violence", "illegal", "weapon", "explosive",
    "hate", "discrimination", "self-harm", "suicide",
]
SIMILARITY_THRESHOLD = 0.8
SELF_IMPROVE_INTERVAL = 100

# Ollama (local LLM) - set by hardware_adapt.detect_and_set_model() before first use
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_PORT = int(os.getenv("OLLAMA_BASE_PORT", "11434"))
# Multi-port URLs for parallel Ollama instances (e.g. 11434, 11435, 11436)
_OLLAMA_PORTS = [OLLAMA_BASE_PORT + i for i in range(PARALLEL_THREADS)]
OLLAMA_URLS = [f"http://localhost:{p}" for p in _OLLAMA_PORTS]

# Obsidian TS Knowledge Vault (visual memory layer)
OBSIDIAN_VAULT = PROJECT_ROOT / os.getenv("OBSIDIAN_VAULT", "obsidian/TS-Knowledge-Vault")

# External knowledge ingestion
ENABLE_EXTERNAL_INGEST = os.getenv("ENABLE_EXTERNAL_INGEST", "false").lower() == "true"
INGEST_RATE_LIMIT_SEC = float(os.getenv("INGEST_RATE_LIMIT_SEC", "2.0"))
INGEST_MAX_RESULTS_PER_QUERY = int(os.getenv("INGEST_MAX_RESULTS_PER_QUERY", "5"))
QUERIES_PATH = DATA_DIR / "queries.json"
QUERY_GENERATE_EVERY_N_CYCLES = int(os.getenv("QUERY_GENERATE_EVERY_N_CYCLES", "3"))

# Obsidian vault analysis
VAULT_MAX_FILES = int(os.getenv("VAULT_MAX_FILES", "200"))

# Vault auto-organization (run when root files > threshold or coherence low)
ORGANIZE_VAULT_ROOT_FILES_THRESHOLD = int(os.getenv("ORGANIZE_VAULT_ROOT_FILES_THRESHOLD", "50"))
ORGANIZE_VAULT_COHERENCE_THRESHOLD = float(os.getenv("ORGANIZE_VAULT_COHERENCE_THRESHOLD", "0.01"))

# Graph snapshots (saved after vault-modifying operations)
SNAPSHOTS_DIR = OBSIDIAN_VAULT / "snapshots"
SNAPSHOT_MAX_WIDTH = int(os.getenv("SNAPSHOT_MAX_WIDTH", "1920"))
SNAPSHOT_MAX_HEIGHT = int(os.getenv("SNAPSHOT_MAX_HEIGHT", "1080"))

# Notifications (optional webhook on failure / change)
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Slack, Discord, or custom webhook
