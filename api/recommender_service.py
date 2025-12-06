from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from src.utils.config import POPULARITY_PATH, USER_HISTORY_PATH
from src.models.baselines import recommend_popularity

class RecommenderService:
    def __init__(self):
        self.popularity = None
        self.user_history = {}

    def load(self):
        if Path(POPULARITY_PATH).exists():
            self.popularity = pd.read_parquet(POPULARITY_PATH)
        if Path(USER_HISTORY_PATH).exists():
            hist_df = pd.read_parquet(USER_HISTORY_PATH)
            # NOTE: use bracket indexing; r["items"] not r.items
            self.user_history = {
                int(r["user_id"]): set(map(int, r["items"]))
                for _, r in hist_df.iterrows()
            }

    def recommend(self, user_id: Optional[int] = None, k: int = 10) -> List[Dict[str, Any]]:
        seen = self.user_history.get(int(user_id), set()) if user_id is not None else set()
        if self.popularity is None or self.popularity.empty:
            return []
        return recommend_popularity(self.popularity, seen, k=k)
