# Optional: LightFM wrapper (hybrid CF)
from __future__ import annotations
import pandas as pd

try:
    from lightfm import LightFM
    from scipy.sparse import coo_matrix
except Exception:  # pragma: no cover
    LightFM = None
    coo_matrix = None

class LightFMRecommender:
    def __init__(self, no_components: int = 64, loss: str = "warp"):
        if LightFM is None:
            raise ImportError("lightfm not available. Install from requirements-optional.txt.")
        self.model = LightFM(no_components=no_components, loss=loss)
        self.user_map = {}
        self.item_map = {}
        self.inv_item_map = {}

    def fit(self, interactions: pd.DataFrame):
        users = interactions["user_id"].astype("category")
        items = interactions["item_id"].astype("category")
        self.user_map = dict(enumerate(users.cat.categories))
        self.item_map = dict(enumerate(items.cat.categories))
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        mat = coo_matrix(
            (interactions["rating"].astype(float),
             (users.cat.codes, items.cat.codes))
        )
        self.model.fit(mat, epochs=10, num_threads=2)

    def recommend_for_user(self, user_id: int, k: int = 10) -> list[int]:
        # map user external id -> internal index
        user_index = None
        for idx, uid in self.user_map.items():
            if int(uid) == int(user_id):
                user_index = idx
                break
        if user_index is None:
            return []
        # score all items for this user
        import numpy as np
        scores = self.model.predict(user_ids=user_index, item_ids=np.arange(len(self.item_map)))
        topk = scores.argsort()[-k:][::-1]
        return [int(self.item_map[i]) for i in topk]
