"""
Phase 2 Configuration
NO TRAINING. Just paths and constants.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRACES_DIR = os.path.join(BASE_DIR, "traces")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# Output paths
PHASE2_OUTPUT_DIR = os.path.join(DATA_DIR, "phase2_evidence")
RETRIEVAL_EVAL_PATH = os.path.join(PHASE2_OUTPUT_DIR, "retrieval_metrics.json")
REACT_EVAL_PATH = os.path.join(PHASE2_OUTPUT_DIR, "react_metrics.json")
PHASE2_OUTPUT_PATH = os.path.join(PHASE2_OUTPUT_DIR, "retrieved_evidence.json")

# Create directories
for d in [DATA_DIR, FAISS_DIR, MODEL_DIR, TRACES_DIR, PROMPTS_DIR,
          os.path.join(DATA_DIR, "scifact"),
          os.path.join(DATA_DIR, "pubmed_abstracts"),
          PHASE2_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Model names
BIOMODEL = "dmis-lab/biobert-base-cased-v1.1"

# OLLAMA CONFIG
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# SciFact paths
SCIFACT_CORPUS = os.path.join(DATA_DIR, "scifact", "corpus.jsonl")
SCIFACT_CLAIMS_TRAIN = os.path.join(DATA_DIR, "scifact", "claims_train.jsonl")
SCIFACT_CLAIMS_TEST = os.path.join(DATA_DIR, "scifact", "claims_test.jsonl")
SCIFACT_CLAIMS_DEV = os.path.join(DATA_DIR, "scifact", "claims_dev.jsonl")

# FAISS index
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "biobert_faiss.index")
FAISS_META_PATH = os.path.join(FAISS_DIR, "abstract_metadata.pkl")

# Retrieval settings
TOP_K_RETRIEVE = 10
MAX_REACT_HOPS = 5

print("Phase 2 Config loaded.")
print(f"Ollama model: {OLLAMA_MODEL}")
print(f"Data directory: {DATA_DIR}")
print(f"FAISS index: {FAISS_INDEX_PATH}")
