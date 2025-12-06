from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

POPULARITY_PATH = ARTIFACTS_DIR / "popularity.parquet"
USER_HISTORY_PATH = ARTIFACTS_DIR / "user_history.parquet"
