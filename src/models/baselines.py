from __future__ import annotations
import pandas as pd

def build_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    # interactions: user_id, item_id, rating, timestamp
    pop = (interactions.groupby("item_id")["rating"]
           .sum()
           .sort_values(ascending=False)
           .reset_index(name="score"))
    rank = pop.assign(rank=range(1, len(pop)+1))
    return rank

def user_history(interactions: pd.DataFrame) -> dict[int, set[int]]:
    d = (interactions.groupby("user_id")["item_id"]
         .apply(set)
         .to_dict())
    return {int(k): set(map(int, v)) for k, v in d.items()}

def recommend_popularity(pop_table: pd.DataFrame, user_seen: set[int] | None, k: int = 10):
    if user_seen:
        filt = pop_table[~pop_table["item_id"].isin(user_seen)]
    else:
        filt = pop_table
    return filt.head(k)[["item_id", "score"]].to_dict(orient="records")
