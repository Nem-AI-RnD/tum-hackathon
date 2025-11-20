import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Determine the environment: "vm" (local machine) or "docker" (container local or remote)
using = "local"
# using = "local-docker"

# Define package root
package_root = Path(__file__).resolve().parents[3]

# Default configurations for "vm" and "docker"
defaults = {
    "local": {
        "CODE_DIR": str(package_root / "src"),
        "DATA_DIR": "",
        "DATA_PKG_DIR": str(package_root / "data"),
        "MODEL_PROVIDER": "google",  # ollama google
    },
    "local-docker": {
        "CODE_DIR": "/app/src/",
        "DATA_DIR": "",
        "DATA_PKG_DIR": "/app/data/",
        "MODEL_PROVIDER": "google",
    },
}

# Apply defaults based on the environment
selected_defaults = defaults[using]
for key, value in selected_defaults.items():
    os.environ.setdefault(key, value)

# Export constants
CODE_DIR = os.environ["CODE_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
DATA_PKG_DIR = os.environ["DATA_PKG_DIR"]
MODEL_PROVIDER = os.environ["MODEL_PROVIDER"]

# GCP settings loaded from .env file
GCP_PROJECT = os.getenv("GCP_PROJECT", "neme-ai-rnd-dev-prj-01")
GCP_FIRESTORE = os.getenv("GCP_FIRESTORE", "ai-assistant-evaluation")
GCP_CS_BUCKET = os.getenv("GCP_CS_BUCKET", "ai-assistant-eval")
GCP_FIRESTORE_COLLECTION = os.getenv("GCP_FIRESTORE_COLLECTION", "Synthesis-Requests")
# LANGFUSE_DATASET_NAME = os.environ["LANGFUSE_DATASET_NAME"]
