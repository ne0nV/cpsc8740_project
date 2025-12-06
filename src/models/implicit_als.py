# Optional: wrapper for implicit library (ALS on implicit feedback)
from __future__ import annotations
import pandas as pd

try:
    import implicit
    from scipy.sparse import coo_matrix
except Exception:  # pragma: no cover
    implicit = None
    coo_matrix = None

class ImplicitALSRecommender:
    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        if implicit is None:
            raise ImportError("implicit not available. Install from requirements-optional.txt.")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        self.user_map = {}
        self.item_map = {}
        self.inv_item_map = {}

    def fit(self, interactions: pd.DataFrame):
        # interactions: user_id,item_id,rating,timestamp
        users = interactions["user_id"].astype("category")
        items = interactions["item_id"].astype("category")
        self.user_map = dict(enumerate(users.cat.categories))
        self.item_map = dict(enumerate(items.cat.categories))
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        mat = coo_matrix(
            (interactions["rating"].astype(float),
             (users.cat.codes, items.cat.codes))
        )
        self.model.fit(mat.T)  # item-user

    def recommend_for_user(self, user_id: int, k: int = 10) -> list[int]:
        if user_id not in self.inv_item_map and user_id not in self.user_map.values():
            return []
        # map user external id -> internal index
        user_index = None
        for idx, uid in self.user_map.items():
            if int(uid) == int(user_id):
                user_index = idx
                break
        if user_index is None:
            return []
        recs = self.model.recommend(user_index, None, N=k)
        # recs: list of (item_index, score)
        return [int(self.item_map[i]) for i, _ in recs]
