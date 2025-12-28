import sys
from pathlib import Path
from typing import Any

# Add parent paths for imports
API_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(API_DIR))

from data.books import DEFAULT_ITEM_COL, DEFAULT_USER_COL, load_pandas_df
from models.mf.model import SVDRecommender

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR: Path = API_DIR / "checkpoints" / "mf"


def demonstrate_predictions(
    model: SVDRecommender,
    user_ids: list,
    item_ids: list,
    n_users: int = 3,
    n_items: int = 2,
) -> None:
    """
    Demonstrate rating predictions for sample user-item pairs.

    Args:
        model: Trained SVDRecommender instance.
        user_ids: list of user IDs to use for demonstration.
        item_ids: list of item IDs to use for demonstration.
        n_users: Number of users to show predictions for.
        n_items: Number of items per user to predict.
    """
    print("\n[Predictions] Sample rating predictions:")
    print("-" * 40)

    for user_id in user_ids[:n_users]:
        for item_id in item_ids[:n_items]:
            try:
                pred = model.predict(user_id, item_id)
                print(f"  User {user_id} -> Item {item_id}: {pred:.2f}")
            except Exception as e:
                print(f"  User {user_id} -> Item {item_id}: Error - {e}")


def demonstrate_recommendations(
    model: SVDRecommender,
    user_ids: list[Any],
    n_users: int = 2,
    n_items: int = 5,
) -> None:
    """
    Demonstrate top-N recommendations for sample users.

    Args:
        model: Trained SVDRecommender instance.
        user_ids: list of user IDs to show recommendations for.
        n_users: Number of users to show recommendations for.
        n_items: Number of items to recommend per user.
    """
    print("\n" + "-" * 40)
    print(f"[Recommendations] Top {n_items} recommendations:")
    print("-" * 40)

    for user_id in user_ids[:n_users]:
        print(f"\n  User {user_id}:")
        try:
            recommendations: list[tuple[Any, float]] = model.recommend(
                user_id, n_items=n_items, exclude_seen=True
            )
            for rank, (item_id, score) in enumerate(recommendations, 1):
                print(f"    {rank}. Item {item_id} (score: {score:.2f})")
        except Exception as e:
            print(f"    Error: {e}")


def main() -> None:
    """
    Main inference pipeline.

    Steps:
        1. Load trained model from checkpoint
        2. Display model metrics
        3. Load sample data
        4. Demonstrate predictions
        5. Demonstrate recommendations
    """
    print("=" * 60)
    print("Matrix Factorization (SVD) Model Inference")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/3] Loading model from checkpoint...")
    model = SVDRecommender()

    try:
        model.load(CHECKPOINT_DIR)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Please run train.py first to create a checkpoint.")
        return

    # Display metrics if available
    metrics = model.load_metrics(CHECKPOINT_DIR)
    if metrics:
        eval_metrics = metrics.get("eval_metrics", {})
        print("  Model metrics:")
        print(f"    - Precision@10: {eval_metrics.get('precision@10', 'N/A'):.4f}")
        print(f"    - Recall@10: {eval_metrics.get('recall@10', 'N/A'):.4f}")

    # Step 2: Load sample data
    print("\n[2/3] Loading sample data...")
    df = load_pandas_df(sample_frac=0.01, min_rating=1)

    sample_users = df[DEFAULT_USER_COL].unique()[:10].tolist()
    sample_items = df[DEFAULT_ITEM_COL].unique()[:10].tolist()

    print(f"  Sample users: {len(sample_users)}")
    print(f"  Sample items: {len(sample_items)}")

    # Step 3: Demonstrate predictions and recommendations
    print("\n[3/3] Running inference demonstrations...")

    demonstrate_predictions(model, sample_users, sample_items)
    demonstrate_recommendations(model, sample_users)

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
