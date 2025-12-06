from __future__ import annotations
import os, sys, json, math
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure we can import local src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils.config import DATA_DIR, ARTIFACTS_DIR
from src.models.metrics import precision_at_k, recall_at_k, ndcg_at_k

# Optional LightFM artifacts (if you unzipped the add-ons)
try:
    from src.utils.config_ext import LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH
    from src.models.serialize import load_lightfm
    HAS_LFM = True
except Exception:
    HAS_LFM = False

K = int(os.getenv("K", "10"))
OUT = ARTIFACTS_DIR / "metrics.json"

def load_dataset() -> tuple[pd.DataFrame, str]:
    ml = DATA_DIR / "ml-100k" / "u.data"
    if ml.exists():
        df = pd.read_csv(ml, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
        return df, "MovieLens-100k"
    sample = DATA_DIR / "sample_interactions.csv"
    if sample.exists():
        return pd.read_csv(sample), "sample_interactions"
    raise SystemExit("No data found. Run scripts/download_movielens.py or ensure data/sample_interactions.csv exists.")

def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp")
    last_idx = df.groupby("user_id")["timestamp"].idxmax()
    test = df.loc[last_idx]
    train = df.drop(last_idx)
    return train, test

def pop_model_recs(train: pd.DataFrame, seen: set[int], k: int) -> list[int]:
    pop = (train.groupby("item_id")["rating"]
                .sum().sort_values(ascending=False).index.tolist())
    return [i for i in pop if i not in seen][:k]

def lfm_model_recs(user_id: int, seen: set[int], k: int, model, user_map: dict[int,int], item_map: dict[int,int]) -> list[int]:
    # user_map: external -> internal, item_map: external -> internal
    if user_id not in user_map:
        return []
    uidx = user_map[user_id]
    n_items = len(item_map)
    inv_item = {v:k for k,v in item_map.items()}
    # Predict all items for this user
    scores = model.predict(user_ids=uidx, item_ids=np.arange(n_items))
    order = scores.argsort()[::-1]
    items = [int(inv_item[i]) for i in order if int(inv_item[i]) not in seen]
    return items[:k]

def evaluate():
    df, name = load_dataset()
    train, test = leave_one_out_split(df)
    user_train = train.groupby("user_id")["item_id"].apply(set).to_dict()
    user_test = test.groupby("user_id")["item_id"].apply(list).to_dict()

    results = {"dataset": name,
               "n_rows": int(len(df)),
               "n_users": int(df["user_id"].nunique()),
               "n_items": int(df["item_id"].nunique()),
               "k": K,
               "models": {}}

    # Baseline popularity
    prec = rec = ndcg = 0.0
    n = 0
    for u, true_list in user_test.items():
        seen = user_train.get(u, set())
        pred = pop_model_recs(train, seen, K)
        prec += precision_at_k(true_list, pred, K)
        rec  += recall_at_k(true_list, pred, K)
        ndcg += ndcg_at_k(true_list, pred, K)
        n += 1
    results["models"]["popularity"] = {
        "precision@k": round(prec/n, 4),
        "recall@k": round(rec/n, 4),
        "ndcg@k": round(ndcg/n, 4)
    }

    # LightFM if available + artifacts present
    if HAS_LFM and Path(LIGHTFM_MODEL_PATH).exists() and Path(LIGHTFM_MAPS_PATH).exists():
        model, user_map, item_map = load_lightfm(LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH)
        prec = rec = ndcg = 0.0
        n = 0
        for u, true_list in user_test.items():
            seen = user_train.get(u, set())
            pred = lfm_model_recs(int(u), seen, K, model, user_map, item_map)
            if not pred:
                # unknown user -> fall back to popularity
                pred = pop_model_recs(train, seen, K)
            prec += precision_at_k(true_list, pred, K)
            rec  += recall_at_k(true_list, pred, K)
            ndcg += ndcg_at_k(true_list, pred, K)
            n += 1
        results["models"]["lightfm"] = {
            "precision@k": round(prec/n, 4),
            "recall@k": round(rec/n, 4),
            "ndcg@k": round(ndcg/n, 4)
        }

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(results, indent=2))
    print(f"Wrote metrics -> {OUT}\n{json.dumps(results, indent=2)}")

if __name__ == "__main__":
    evaluate()
