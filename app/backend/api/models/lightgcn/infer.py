import sys
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(API_DIR))

from models.lightgcn.model import LightGCNRecommender

CHECKPOINT_DIR: Path = API_DIR / "checkpoints" / "lightgcn"


def demonstrate_predictions(
    model: LightGCNRecommender,
    user_ids: list,
    item_ids: list,
    n_users: int = 3,
    n_items: int = 2,
) -> None:
    """Demonstrate score predictions."""
    print("\n[Predictions] Sample scores:")
    print("-" * 40)

    known_users = [u for u in user_ids if u in model.user2idx][:n_users]
    known_items = [i for i in item_ids if i in model.item2idx][:n_items]

    if not known_users:
        print("  No known users found.")
        return

    for user_id in known_users:
        for item_id in known_items:
            try:
                score = model.predict(user_id, item_id)
                print(f"  User {user_id} -> Item {item_id}: {score:.4f}")
            except Exception as e:
                print(f"  User {user_id} -> Item {item_id}: Error - {e}")


def demonstrate_recommendations(
    model: LightGCNRecommender,
    user_ids: list,
    n_users: int = 2,
    n_items: int = 5,
) -> None:
    """Demonstrate top-N recommendations."""
    print("\n" + "-" * 40)
    print(f"[Recommendations] Top {n_items} recommendations:")
    print("-" * 40)

    known_users = [u for u in user_ids if u in model.user2idx][:n_users]

    if not known_users:
        print("  No known users found.")
        return

    for user_id in known_users:
        print(f"\n  User {user_id}:")
        try:
            recommendations: list[tuple[Any, float]] = model.recommend(
                user_id, n_items=n_items, exclude_seen=True
            )
            for rank, (item_id, score) in enumerate(recommendations, 1):
                print(f"    {rank}. Item {item_id} (score: {score:.4f})")
        except Exception as e:
            print(f"    Error: {e}")


def main() -> None:
    """Main inference pipeline."""
    print("=" * 60)
    print("LightGCN Model Inference")
    print("=" * 60)

    print("\n[1/3] Loading model from checkpoint...")
    model = LightGCNRecommender()

    try:
        model.load(CHECKPOINT_DIR)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Please run train.py first.")
        return

    metrics = model.load_metrics(CHECKPOINT_DIR)
    if metrics:
        eval_metrics = metrics.get("eval_metrics", {})
        print("  Model metrics:")
        print(f"    - Precision@10: {eval_metrics.get('precision@10', 'N/A'):.4f}")
        print(f"    - Recall@10: {eval_metrics.get('recall@10', 'N/A'):.4f}")

    print("\n[2/3] Getting known users/items from model...")
    sample_users = list(model.user2idx.keys())[:20]
    sample_items = list(model.item2idx.keys())[:20]

    print(f"  Known users in model: {len(model.user2idx)}")
    print(f"  Known items in model: {len(model.item2idx)}")
    print(f"  Sample users: {len(sample_users)}")
    print(f"  Sample items: {len(sample_items)}")

    print("\n[3/3] Running inference demonstrations...")
    demonstrate_predictions(model, sample_users, sample_items)
    demonstrate_recommendations(model, sample_users)

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
