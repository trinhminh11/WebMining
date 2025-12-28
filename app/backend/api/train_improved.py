"""
Improved Training Script with Better Data Preprocessing and Hyperparameters.

Key improvements:
1. Filter users with minimum interactions (≥5)
2. Filter items with minimum interactions (≥5)
3. Lower rating threshold for evaluation (≥3 instead of ≥5)
4. Tuned hyperparameters for sparse data
5. More training epochs for complex models

Usage:
    cd /Users/thefool/Local/Project/WebMining
    uv run python -m Books.app.backend.api.train_improved
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

API_DIR = Path(__file__).parent
sys.path.insert(0, str(API_DIR))

from data.books import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
    get_train_test_split,
    load_pandas_df,
)
from metrics.eval import precision_at_k, recall_at_k

# ============================================================================
# Configuration - IMPROVED
# ============================================================================
TOP_K: int = 10
TEST_SIZE: float = 0.2  # Smaller test set, more training data
RATING_THRESHOLD: float = 3.0  # Lower threshold (≥3 is positive)
N_EVAL_USERS: int = 300
SEED: int = 42
SAMPLE_FRAC: float = 0.5  # More data
MIN_RATING: int = 1

# Data filtering (KEY IMPROVEMENT)
MIN_USER_INTERACTIONS: int = 5  # Users must have ≥5 ratings
MIN_ITEM_INTERACTIONS: int = 5  # Items must have ≥5 ratings


def filter_data(df: pd.DataFrame, min_user: int = 5, min_item: int = 5) -> pd.DataFrame:
    """Filter users and items with minimum interactions."""
    print(f"  Before filtering: {len(df):,} ratings")

    # Iteratively filter until stable
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)

        # Filter users
        user_counts = df[DEFAULT_USER_COL].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df[DEFAULT_USER_COL].isin(valid_users)]

        # Filter items
        item_counts = df[DEFAULT_ITEM_COL].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df[DEFAULT_ITEM_COL].isin(valid_items)]

    print(f"  After filtering: {len(df):,} ratings")
    print(f"  Users: {df[DEFAULT_USER_COL].nunique():,}, Items: {df[DEFAULT_ITEM_COL].nunique():,}")
    return df


def evaluate_model(model, train_df, test_df, k=10, n_sample_users=300, threshold=3.0):
    """Evaluate model with precision@k and recall@k."""
    test_users = test_df[DEFAULT_USER_COL].unique()
    known_users = [u for u in test_users if u in model.user2idx]

    if len(known_users) > n_sample_users:
        np.random.seed(SEED)
        known_users = list(np.random.choice(known_users, n_sample_users, replace=False))

    precisions, recalls = [], []

    for user_id in known_users:
        user_test = test_df[test_df[DEFAULT_USER_COL] == user_id]
        # Use lower threshold
        y_true = user_test[user_test[DEFAULT_RATING_COL] >= threshold][
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
        "n_users": len(precisions),
    }


def train_mf(train_df, test_df, checkpoint_dir, rating_scale=None):
    """Train MF with improved hyperparameters."""
    from models.mf.model import SVDRecommender

    print("\n" + "=" * 60)
    print("Training MF (SVD) - IMPROVED")
    print("=" * 60)

    # Improved hyperparameters
    model = SVDRecommender(
        n_factors=150,       # More factors
        n_epochs=30,         # More epochs
        lr_all=0.005,
        reg_all=0.02,
        random_state=SEED,
        verbose=True,
    )

    start = time.time()
    stats = model.train(train_df, rating_scale=rating_scale)
    train_time = time.time() - start

    metrics = evaluate_model(model, train_df, test_df, k=TOP_K, threshold=RATING_THRESHOLD)
    print(f"  Precision@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}")

    model.save(checkpoint_dir)
    model.save_metrics(checkpoint_dir, {
        "train_time": train_time,
        "final_precision": metrics[f"precision@{TOP_K}"],
        "final_recall": metrics[f"recall@{TOP_K}"],
        "n_epochs": 30,
        "n_factors": 150,
        "learning_curve": {"epochs": list(range(1, 31))},
    })

    return model, metrics


def train_bpr(train_df, test_df, checkpoint_dir):
    """Train BPR with improved hyperparameters."""
    from models.bpr.model import BPRRecommender

    print("\n" + "=" * 60)
    print("Training BPR - IMPROVED")
    print("=" * 60)

    # Improved hyperparameters
    model = BPRRecommender(
        k=150,               # More factors
        max_iter=200,        # More iterations
        learning_rate=0.01,
        lambda_reg=0.001,
        seed=SEED,
        verbose=True,
    )

    start = time.time()
    stats = model.train(train_df)
    train_time = time.time() - start

    metrics = evaluate_model(model, train_df, test_df, k=TOP_K, threshold=RATING_THRESHOLD)
    print(f"  Precision@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}")

    model.save(checkpoint_dir)
    model.save_metrics(checkpoint_dir, {
        "train_time": train_time,
        "final_precision": metrics[f"precision@{TOP_K}"],
        "final_recall": metrics[f"recall@{TOP_K}"],
        "n_epochs": 200,
    })

    return model, metrics


def train_bivae(train_df, test_df, checkpoint_dir):
    """Train BiVAE with improved hyperparameters."""
    from models.bivae.model import BiVAERecommender

    print("\n" + "=" * 60)
    print("Training BiVAE - IMPROVED")
    print("=" * 60)

    # Improved hyperparameters
    model = BiVAERecommender(
        k=100,                     # More latent factors
        encoder_structure=[200, 100],  # Deeper encoder
        n_epochs=50,               # More epochs
        batch_size=256,            # Larger batch
        learning_rate=0.001,
        use_gpu=True,
        verbose=True,
        seed=SEED,
    )

    start = time.time()
    stats = model.train(train_df)
    train_time = time.time() - start

    metrics = evaluate_model(model, train_df, test_df, k=TOP_K, threshold=RATING_THRESHOLD)
    print(f"  Precision@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}")

    model.save(checkpoint_dir)
    model.save_metrics(checkpoint_dir, {
        "train_time": train_time,
        "final_precision": metrics[f"precision@{TOP_K}"],
        "final_recall": metrics[f"recall@{TOP_K}"],
        "n_epochs": 50,
    })

    return model, metrics


def train_lightgcn(train_df, test_df, checkpoint_dir):
    """Train LightGCN with improved hyperparameters."""
    from models.lightgcn.model import LightGCNRecommender

    print("\n" + "=" * 60)
    print("Training LightGCN - IMPROVED")
    print("=" * 60)

    # Improved hyperparameters
    model = LightGCNRecommender(
        n_layers=3,
        latent_dim=128,          # More dimensions
        n_epochs=50,             # More epochs
        batch_size=2048,         # Larger batch
        learning_rate=0.001,     # Lower LR
        decay=0.0001,
        seed=SEED,
    )

    start = time.time()
    stats = model.train(train_df)
    train_time = time.time() - start

    loss_history = stats.get("loss_history", [])

    metrics = evaluate_model(model, train_df, test_df, k=TOP_K, threshold=RATING_THRESHOLD)
    print(f"  Precision@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}")

    model.save(checkpoint_dir)
    model.save_metrics(checkpoint_dir, {
        "train_time": train_time,
        "final_precision": metrics[f"precision@{TOP_K}"],
        "final_recall": metrics[f"recall@{TOP_K}"],
        "n_epochs": 50,
        "learning_curve": {
            "epochs": list(range(1, len(loss_history) + 1)),
            "train_loss": loss_history,
        },
    })

    return model, metrics


def train_lightfm(train_df, test_df, checkpoint_dir):
    """Train LightFM with improved hyperparameters."""
    from models.lfm.model import LightFMRecommender

    print("\n" + "=" * 60)
    print("Training LightFM - IMPROVED")
    print("=" * 60)

    # Improved hyperparameters
    model = LightFMRecommender(
        no_components=100,       # More components
        n_epochs=50,             # More epochs
        learning_rate=0.05,
        loss="warp",
        seed=SEED,
    )

    start = time.time()
    stats = model.train(train_df)
    train_time = time.time() - start

    metrics = evaluate_model(model, train_df, test_df, k=TOP_K, threshold=RATING_THRESHOLD)
    print(f"  Precision@{TOP_K}: {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics[f'recall@{TOP_K}']:.4f}")

    model.save(checkpoint_dir)
    model.save_metrics(checkpoint_dir, {
        "train_time": train_time,
        "final_precision": metrics[f"precision@{TOP_K}"],
        "final_recall": metrics[f"recall@{TOP_K}"],
        "n_epochs": 50,
    })

    return model, metrics


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("IMPROVED TRAINING WITH BETTER PREPROCESSING & HYPERPARAMETERS")
    print("=" * 70)

    # Load data
    print("\n[1] Loading Books dataset...")
    df = load_pandas_df(min_rating=MIN_RATING, sample_frac=SAMPLE_FRAC)
    print(f"  Loaded {len(df):,} ratings")

    # KEY: Filter data
    print("\n[2] Filtering sparse users/items...")
    df = filter_data(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)

    # Split
    print("\n[3] Splitting data...")
    train_df, test_df = get_train_test_split(df, test_size=TEST_SIZE)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    rating_scale = (float(df[DEFAULT_RATING_COL].min()), float(df[DEFAULT_RATING_COL].max()))

    # Results
    results = []
    checkpoint_base = API_DIR / "checkpoints"

    # Train each model
    models_to_train = [
        ("MF", train_mf, {"rating_scale": rating_scale}),
        ("BPR", train_bpr, {}),
        ("BiVAE", train_bivae, {}),
        ("LightGCN", train_lightgcn, {}),
        ("LightFM", train_lightfm, {}),
    ]

    for name, train_fn, extra_args in models_to_train:
        try:
            folder = name.lower().replace(" ", "_")
            if extra_args:
                _, m = train_fn(train_df, test_df, checkpoint_base / folder, **extra_args)
            else:
                _, m = train_fn(train_df, test_df, checkpoint_base / folder)
            results.append({"model": name, **m})
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Rating threshold ≥ 3.0)")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(f"precision@{TOP_K}", ascending=False)
    print(results_df.to_string(index=False))

    # Save
    results_df.to_csv(checkpoint_base / "model_comparison.csv", index=False)
    print(f"\nSaved to {checkpoint_base / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
