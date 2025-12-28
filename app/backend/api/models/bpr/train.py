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
from models.bpr.model import BPRRecommender

# ============================================================================
# Configuration
# ============================================================================
TOP_K: int = 10
TEST_SIZE: float = 0.25
CHECKPOINT_DIR: Path = API_DIR / "checkpoints" / "bpr"

# Model hyperparameters
K: int = 100  # Latent factors
MAX_ITER: int = 100
LEARNING_RATE: float = 0.01
LAMBDA_REG: float = 0.001

# Data sampling
SAMPLE_FRAC: float = 0.1
MIN_RATING: int = 1


def evaluate_model(
    model: BPRRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    n_sample_users: int = 100,
    rating_threshold: float = 5.0,
) -> dict[str, float]:
    """
    Evaluate the BPR model using precision@k and recall@k.

    Args:
        model: Trained BPRRecommender instance.
        train_df: Training data.
        test_df: Test data.
        k: Number of items for metrics calculation.
        n_sample_users: Maximum users to evaluate.
        rating_threshold: Min rating for "relevant" items.

    Returns:
        Dictionary with metrics.
    """
    test_users = test_df[DEFAULT_USER_COL].unique()
    known_users = [u for u in test_users if u in model.user2idx]

    if len(known_users) > n_sample_users:
        np.random.seed(42)
        known_users = list(np.random.choice(known_users, n_sample_users, replace=False))

    precisions: list[float] = []
    recalls: list[float] = []

    for user_id in known_users:
        user_test = test_df[test_df[DEFAULT_USER_COL] == user_id]
        y_true = user_test[user_test[DEFAULT_RATING_COL] >= rating_threshold][
            DEFAULT_ITEM_COL
        ].values

        y_true = [it for it in y_true if it in model.item2idx]

        if len(y_true) == 0:
            continue

        try:
            recommendations = model.recommend(user_id, n_items=k, exclude_seen=True)
            y_pred = np.array([item_id for item_id, _ in recommendations])
        except Exception:
            continue

        precisions.append(precision_at_k(np.array(y_true), y_pred, k=k))
        recalls.append(recall_at_k(np.array(y_true), y_pred, k=k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_evaluated_users": len(precisions),
    }


def main() -> None:
    """Main training pipeline for BPR model."""
    print("=" * 60)
    print("Bayesian Personalized Ranking (BPR) Model Training")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading Books dataset...")
    start_time = time.time()

    df = load_pandas_df(min_rating=MIN_RATING, sample_frac=SAMPLE_FRAC)
    print(f"  Loaded {len(df):,} ratings")
    print(f"  Users: {df[DEFAULT_USER_COL].nunique():,}")
    print(f"  Items: {df[DEFAULT_ITEM_COL].nunique():,}")
    print(f"  Time: {time.time() - start_time:.2f}s")

    # Split data
    print(f"\n[2/5] Splitting data (test_size={TEST_SIZE})...")
    train_df, test_df = get_train_test_split(df, test_size=TEST_SIZE)
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")

    # Initialize model
    print("\n[3/5] Initializing BPR model...")
    print(f"  k={K}, max_iter={MAX_ITER}")

    model = BPRRecommender(
        k=K,
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        verbose=True,
        seed=42,
    )

    # Train
    print("\n[4/5] Training model...")
    start_time = time.time()

    train_stats = model.train(train_df)
    train_time = time.time() - start_time

    print(f"  Training completed in {train_time:.2f}s")
    print(f"  Trainset: {train_stats['n_users']} users, {train_stats['n_items']} items")

    # Evaluate
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
            "k": K,
            "max_iter": MAX_ITER,
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
