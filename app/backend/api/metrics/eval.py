import numpy as np

def recall_at_k(y_true: np.array, y_pred: np.array, k=10):
    """
    Calculates recall@k.

    Args:
        y_true (np.array): List of ground truth items.
        y_pred (np.array): List of predicted items (ranked).
        k (int): Top k items to consider.

    Returns:
        float: Recall score.
    """
    if len(y_true) == 0:
        return 0.0

    y_pred_k = y_pred[:k]

    # Calculate intersection
    relevant_retrieved = len(set(y_true) & set(y_pred_k))

    return relevant_retrieved / len(y_true)

def precision_at_k(y_true: np.array, y_pred: np.array, k=10):
    """
    Calculates precision@k.

    Args:
        y_true (np.array): List of ground truth items.
        y_pred (np.array): List of predicted items (ranked).
        k (int): Top k items to consider.

    Returns:
        float: Precision score.
    """
    if k == 0:
        return 0.0

    y_pred_k = y_pred[:k]

    # Calculate intersection
    relevant_retrieved = len(set(y_true) & set(y_pred_k))

    return relevant_retrieved / k
