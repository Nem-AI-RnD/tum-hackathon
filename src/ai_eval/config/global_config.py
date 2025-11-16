import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Determine the environment: "vm" (local machine) or "docker" (container local or remote)
using = "vm"
# using = "docker"

# Define package root
package_root = Path(__file__).resolve().parents[3]

# Default configurations for "vm" and "docker"
defaults = {
    "vm": {
        "CODE_DIR": str(package_root / "src"),
        "DATA_DIR": "",
        "DATA_PKG_DIR": str(package_root / "data"),
        "GCP_PROJECT": "neme-ai-rnd-dev-prj-01",
        "GCP_FIRESTORE": "ai-assistant-evaluation",
        "GCP_FIRESTORE_COLLECTION": "Synthesis-Requests",
        "GCP_CS_BUCKET": "ai-assistant-eval",
        # "LANGFUSE_DATASET_NAME": "nemyevaluation",
        "MODEL_PROVIDER": "google",  # ollama google
    },
    "docker": {
        "CODE_DIR": "/app/src/",
        "DATA_DIR": "",
        "DATA_PKG_DIR": "/app/data/",
        "GCP_PROJECT": "neme-ai-rnd-dev-prj-01",
        "GCP_FIRESTORE": "ai-assistant-evaluation",
        "GCP_FIRESTORE_COLLECTION": "Synthesis-Requests",
        "GCP_CS_BUCKET": "ai-assistant-eval",
        # "LANGFUSE_DATASET_NAME": "nemyevaluation",
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
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_FIRESTORE = os.environ["GCP_FIRESTORE"]
GCP_CS_BUCKET = os.environ["GCP_CS_BUCKET"]
MODEL_PROVIDER = os.environ["MODEL_PROVIDER"]
GCP_FIRESTORE_COLLECTION = os.environ["GCP_FIRESTORE_COLLECTION"]
# LANGFUSE_DATASET_NAME = os.environ["LANGFUSE_DATASET_NAME"]
