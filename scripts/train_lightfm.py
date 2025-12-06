from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils.config import DATA_DIR
from src.utils.config_ext import LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH
from src.models.serialize import save_lightfm

# Training LightFM requires 'lightfm' and 'scipy' installed.
# Recommended on Apple Silicon:
#   conda install -c conda-forge lightfm scipy numpy

def load_interactions() -> pd.DataFrame:
    # Prefer MovieLens if present, else fall back to sample_interactions.csv
    ml_dir = DATA_DIR / "ml-100k"
    if (ml_dir / "u.data").exists() and (ml_dir / "u.item").exists():
        # Load u.data (user item rating timestamp)
        df = pd.read_csv(ml_dir / "u.data", sep="\t", names=["user_id","item_id","rating","timestamp"])
        return df
    sample = DATA_DIR / "sample_interactions.csv"
    if sample.exists():
        return pd.read_csv(sample)
    raise SystemExit("No data found. Run `python scripts/download_movielens.py` or ensure data/sample_interactions.csv exists.")

def train_lightfm_model(df: pd.DataFrame, no_components: int = 64, loss: str = "warp"):
    from lightfm import LightFM
    from scipy.sparse import coo_matrix

    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    user_map = dict(enumerate(users.cat.categories))
    item_map = dict(enumerate(items.cat.categories))
    inv_user = {int(v): int(k) for k, v in user_map.items()}
    inv_item = {int(v): int(k) for k, v in item_map.items()}

    mat = coo_matrix((df["rating"].astype(float), (users.cat.codes, items.cat.codes)))
    model = LightFM(no_components=no_components, loss=loss)
    model.fit(mat, epochs=15, num_threads=2)
    return model, inv_user, inv_item

def main():
    df = load_interactions()
    print(f"Loaded interactions: {len(df):,} rows")
    model, user_map, item_map = train_lightfm_model(df)
    save_lightfm(model, user_map, item_map, LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH)
    print(f"Saved LightFM model -> {LIGHTFM_MODEL_PATH}")
    print(f"Saved maps -> {LIGHTFM_MAPS_PATH}")

if __name__ == "__main__":
    main()
