from __future__ import annotations
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_DIR, ARTIFACTS_DIR, POPULARITY_PATH, USER_HISTORY_PATH
from src.models.baselines import build_popularity, user_history
from src.utils.io import write_parquet

def main():
    sample = DATA_DIR / "sample_interactions.csv"
    if not sample.exists():
        raise SystemExit(f"Missing {sample}.")
    df = pd.read_csv(sample)
    pop = build_popularity(df)
    write_parquet(pop, POPULARITY_PATH)
    hist = user_history(df)
    # save history as two-column parquet for simplicity
    rows = [{"user_id": u, "items": list(s)} for u, s in hist.items()]
    write_parquet(pd.DataFrame(rows), USER_HISTORY_PATH)
    print(f"Saved popularity -> {POPULARITY_PATH}")
    print(f"Saved user history -> {USER_HISTORY_PATH}")

if __name__ == "__main__":
    main()
