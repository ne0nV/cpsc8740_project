from __future__ import annotations
import numpy as np

def precision_at_k(y_true: list[int], y_pred: list[int], k: int = 10) -> float:
    if k == 0:
        return 0.0
    topk = set(y_pred[:k])
    return len(topk.intersection(y_true)) / float(k)

def recall_at_k(y_true: list[int], y_pred: list[int], k: int = 10) -> float:
    denom = len(set(y_true))
    if denom == 0:
        return 0.0
    topk = set(y_pred[:k])
    return len(topk.intersection(y_true)) / float(denom)

def ndcg_at_k(y_true: list[int], y_pred: list[int], k: int = 10) -> float:
    # binary relevance
    dcg = 0.0
    for i, item in enumerate(y_pred[:k]):
        rel = 1.0 if item in y_true else 0.0
        dcg += rel / np.log2(i + 2)  # positions start at 1
    # ideal DCG
    ideal_rel = [1.0] * min(len(set(y_true)), k)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rel))
    return float(dcg / idcg) if idcg > 0 else 0.0
