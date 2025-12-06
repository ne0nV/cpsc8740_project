from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from src.utils.config import POPULARITY_PATH, USER_HISTORY_PATH
from src.utils.config_ext import LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH
from src.models.baselines import recommend_popularity

class RecommenderServiceLFM:
    def __init__(self):
        self.popularity = None
        self.user_history = {}
        self.lfm = None
        self.user_map = {}
        self.item_map = {}

    def load(self):
        # Load baseline artifacts
        if Path(POPULARITY_PATH).exists():
            self.popularity = pd.read_parquet(POPULARITY_PATH)
        if Path(USER_HISTORY_PATH).exists():
            hist_df = pd.read_parquet(USER_HISTORY_PATH)
            records = hist_df.to_dict(orient="records")
            self.user_history = {
                int(rec["user_id"]): {int(x) for x in rec["items"]}
                for rec in records
}

        # Load LightFM if available
        try:
            from src.models.serialize import load_lightfm
            if Path(LIGHTFM_MODEL_PATH).exists() and Path(LIGHTFM_MAPS_PATH).exists():
                self.lfm, self.user_map, self.item_map = load_lightfm(LIGHTFM_MODEL_PATH, LIGHTFM_MAPS_PATH)
        except Exception as e:  # keep API alive even if LFM fails
            print("LightFM load failed:", e)

    def _lfm_recommend(self, user_id: int, k: int = 10) -> List[int]:
        # Map external user id -> internal index
        if not self.lfm or user_id not in self.user_map:
            return []
        uidx = self.user_map[user_id]
        scores = self.lfm.predict(user_ids=uidx, item_ids=np.arange(len(self.item_map)))
        topk = scores.argsort()[-k:][::-1]
        # map back to external ids by inverting item_map
        inv_item = {v: k for k, v in self.item_map.items()}
        return [int(inv_item[i]) for i in topk]

    def recommend(self, user_id: Optional[int] = None, k: int = 10) -> List[Dict[str, Any]]:
        seen = self.user_history.get(int(user_id), set()) if user_id is not None else set()
        # Prefer LFM if available and user known
        if self.lfm and user_id is not None and int(user_id) in self.user_map:
            items = [i for i in self._lfm_recommend(int(user_id), k=k + len(seen)) if i not in seen][:k]
            return [{"item_id": it, "score": float(idx)} for idx, it in enumerate(items, 1)]
        # Fallback to popularity
        if self.popularity is None or self.popularity.empty:
            return []
        return recommend_popularity(self.popularity, seen, k=k)
