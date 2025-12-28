import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent paths for imports
API_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(API_DIR))

from data.books import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
    get_train_test_split,
    load_pandas_df,
)
from metrics.eval import precision_at_k, recall_at_k
from models.mf.model import SVDRecommender

# ============================================================================
# Configuration
# ============================================================================
TOP_K: int = 10
TEST_SIZE: float = 0.25
CHECKPOINT_DIR: Path = API_DIR / "checkpoints" / "mf"

# Model hyperparameters
N_FACTORS: int = 100
N_EPOCHS: int = 20
LEARNING_RATE: float = 0.005
REGULARIZATION: float = 0.02

# Data sampling (for faster training during development)
SAMPLE_FRAC: float = 0.1  # Use 10% of data
MIN_RATING: int = 1  # Filter out zero ratings


def evaluate_model(
    model: SVDRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    n_sample_users: int = 100,
    rating_threshold: float = 5.0,
) -> dict[str, float]:
    """
    Evaluate the recommendation model using precision@k and recall@k.

    For each user, ground truth is defined as items rated >= rating_threshold
    in the test set. Predictions are the top-k recommended items.

    Args:
        model: Trained SVDRecommender instance.
        train_df: Training data (used to exclude seen items).
        test_df: Test data containing ground truth ratings.
        k: Number of items for precision/recall calculation.
        n_sample_users: Maximum number of users to evaluate (for speed).
        rating_threshold: Minimum rating to consider an item as "relevant".

    Returns:
        Dictionary containing:
            - precision@k: Mean precision across evaluated users
            - recall@k: Mean recall across evaluated users
            - n_evaluated_users: Number of users actually evaluated
    """
    # Get unique users in test set
    test_users = test_df[DEFAULT_USER_COL].unique()

    # Sample users if too many
    if len(test_users) > n_sample_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, n_sample_users, replace=False)

    precisions: list[float] = []
    recalls: list[float] = []

    for user_id in test_users:
        # Get ground truth: items user rated highly in test set
        user_test = test_df[test_df[DEFAULT_USER_COL] == user_id]
        y_true = user_test[user_test[DEFAULT_RATING_COL] >= rating_threshold][
            DEFAULT_ITEM_COL
        ].values

        if len(y_true) == 0:
            continue

        # Get recommendations
        try:
            recommendations = model.recommend(user_id, n_items=k, exclude_seen=True)
            y_pred = np.array([item_id for item_id, _ in recommendations])
        except Exception:
            continue

        # Calculate metrics
        precisions.append(precision_at_k(y_true, y_pred, k=k))
        recalls.append(recall_at_k(y_true, y_pred, k=k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_evaluated_users": len(precisions),
    }


def main() -> None:
    """
    Main training pipeline.

    Steps:
        1. Load and preprocess Books dataset
        2. Split into train/test sets
        3. Train SVD model
        4. Evaluate on test set
        5. Save model checkpoint and metrics
    """
    print("=" * 60)
    print("Matrix Factorization (SVD) Model Training")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading Books dataset...")
    start_time = time.time()

    df = load_pandas_df(min_rating=MIN_RATING, sample_frac=SAMPLE_FRAC)
    print(f"  Loaded {len(df):,} ratings")
    print(f"  Users: {df[DEFAULT_USER_COL].nunique():,}")
    print(f"  Items: {df[DEFAULT_ITEM_COL].nunique():,}")
    print(f"  Time: {time.time() - start_time:.2f}s")

    # Step 2: Split data
    print(f"\n[2/5] Splitting data (test_size={TEST_SIZE})...")
    train_df, test_df = get_train_test_split(df, test_size=TEST_SIZE)
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")

    # Step 3: Initialize model
    print("\n[3/5] Initializing SVD model...")
    print(f"  n_factors={N_FACTORS}, n_epochs={N_EPOCHS}")
    model = SVDRecommender(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        lr_all=LEARNING_RATE,
        reg_all=REGULARIZATION,
        verbose=True,
    )

    # Step 4: Train
    print("\n[4/5] Training model...")
    start_time = time.time()

    rating_min = float(df[DEFAULT_RATING_COL].min())
    rating_max = float(df[DEFAULT_RATING_COL].max())

    train_stats = model.train(train_df, rating_scale=(rating_min, rating_max))
    train_time = time.time() - start_time

    print(f"  Training completed in {train_time:.2f}s")
    print(f"  Trainset: {train_stats['n_users']} users, {train_stats['n_items']} items")

    # Step 5: Evaluate
    print(f"\n[5/5] Evaluating model (top-{TOP_K})...")
    start_time = time.time()
    eval_metrics = evaluate_model(model, train_df, test_df, k=TOP_K)
    eval_time = time.time() - start_time

    print(f"  Precision@{TOP_K}: {eval_metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {eval_metrics[f'recall@{TOP_K}']:.4f}")
    print(f"  Evaluated {eval_metrics['n_evaluated_users']} users in {eval_time:.2f}s")

    # Save model and metrics
    print(f"\n[Saving] Saving model to {CHECKPOINT_DIR}...")
    model.save(CHECKPOINT_DIR)

    metrics = {
        "train_history": {
            "n_factors": N_FACTORS,
            "n_epochs": N_EPOCHS,
            "train_time_seconds": train_time,
            "n_train_ratings": len(train_df),
            "n_users": train_stats["n_users"],
            "n_items": train_stats["n_items"],
        },
        "eval_metrics": {
            f"precision@{TOP_K}": eval_metrics[f"precision@{TOP_K}"],
            f"recall@{TOP_K}": eval_metrics[f"recall@{TOP_K}"],
            "n_evaluated_users": eval_metrics["n_evaluated_users"],
        },
    }
    model.save_metrics(CHECKPOINT_DIR, metrics)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
