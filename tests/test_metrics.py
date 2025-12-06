from src.models.metrics import precision_at_k, recall_at_k, ndcg_at_k

def test_metrics_simple():
    y_true = [1,2,3]
    y_pred = [3,4,5,1]
    assert 0 <= precision_at_k(y_true, y_pred, k=3) <= 1
    assert 0 <= recall_at_k(y_true, y_pred, k=3) <= 1
    assert 0 <= ndcg_at_k(y_true, y_pred, k=3) <= 1
