from pathlib import Path

# The project root is defined as the parent of the 'src' directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Results Paths
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
MODELS = PROJECT_ROOT / "models"

# Ensure directories exist upon import
for path in [DATA_RAW, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_LOGS, MODELS]:
    path.mkdir(parents=True, exist_ok=True)