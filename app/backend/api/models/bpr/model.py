import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import BaseRecommender


class BPRRecommender(BaseRecommender):
    """
    BPR-based recommender using the Cornac library.

    This model uses Bayesian Personalized Ranking with matrix factorization
    to learn pairwise preferences from implicit feedback data.

    Attributes:
        k: Number of latent factors.
        max_iter: Maximum number of training iterations.
        learning_rate: Learning rate for SGD.
        lambda_reg: Regularization coefficient.
        verbose: Whether to print training progress.
        seed: Random seed for reproducibility.
        model: The trained Cornac BPR model.
        train_set: The Cornac Dataset object.
        user2idx: Mapping from original user IDs to Cornac indices.
        idx2user: Mapping from Cornac indices to original user IDs.
        item2idx: Mapping from original item IDs to Cornac indices.
        idx2item: Mapping from Cornac indices to original item IDs.

    Example:
        >>> model = BPRRecommender(k=100, max_iter=100)
        >>> stats = model.train(train_df)
        >>> recs = model.recommend(user_id=123, n_items=10)
    """

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.001,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the BPR recommender.

        Args:
            k: Number of latent factors. Higher values capture more
                complex user-item interactions.
            max_iter: Maximum number of SGD iterations.
            learning_rate: Learning rate for stochastic gradient descent.
            lambda_reg: L2 regularization coefficient.
            verbose: If True, print training progress with a progress bar.
            seed: Random seed for reproducibility.
        """
        super().__init__(name="bpr")
        self.k: int = k
        self.max_iter: int = max_iter
        self.learning_rate: float = learning_rate
        self.lambda_reg: float = lambda_reg
        self.verbose: bool = verbose
        self.seed: int = seed

        self.model = None
        self.train_set = None
        self.train_df: Optional[pd.DataFrame] = None

        # ID mappings (Cornac uses internal indices)
        self.user2idx: dict[Any, int] = {}
        self.idx2user: dict[int, Any] = {}
        self.item2idx: dict[Any, int] = {}
        self.idx2item: dict[int, Any] = {}

        # Column names
        self.user_col: str = "userID"
        self.item_col: str = "itemID"
        self.rating_col: str = "rating"

    def train(
        self,
        train_data: pd.DataFrame,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Train the BPR model.

        Args:
            train_data: DataFrame with userID, itemID, rating columns.
                Ratings are used as implicit feedback (any rating = interaction).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary containing training statistics.

        Raises:
            ImportError: If cornac library is not installed.
        """
        try:
            import cornac
        except ImportError:
            raise ImportError(
                "BPR requires the cornac library. Install with: pip install cornac"
            )

        self.train_df = train_data.copy()

        # Create Cornac dataset from user-item-rating tuples
        # Cornac expects an iterable of (user, item, rating) tuples
        uir_tuples = list(train_data[[
            self.user_col, self.item_col, self.rating_col
        ]].itertuples(index=False, name=None))

        self.train_set = cornac.data.Dataset.from_uir(
            uir_tuples,
            seed=self.seed
        )

        # Store ID mappings from Cornac
        self.user2idx = self.train_set.uid_map
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.item2idx = self.train_set.iid_map
        self.idx2item = {v: k for k, v in self.item2idx.items()}

        # Initialize BPR model
        self.model = cornac.models.BPR(
            k=self.k,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            verbose=self.verbose,
            seed=self.seed
        )

        # Train model
        self.model.fit(self.train_set)
        self.is_trained = True

        return {
            "n_users": self.train_set.num_users,
            "n_items": self.train_set.num_items,
            "k": self.k,
            "max_iter": self.max_iter,
        }

    def predict(self, user_id: Any, item_id: Any) -> float:
        """
        Predict the preference score for a user-item pair.

        Args:
            user_id: Original user ID.
            item_id: Original item ID.

        Returns:
            Predicted preference score.

        Raises:
            ValueError: If model not trained or IDs unknown.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if user_id not in self.user2idx:
            raise ValueError(f"Unknown user_id: {user_id}")
        if item_id not in self.item2idx:
            raise ValueError(f"Unknown item_id: {item_id}")

        user_idx = self.user2idx[user_id]
        item_idx = self.item2idx[item_id]

        score = self.model.score(user_idx, item_idx)
        return float(score)

    def recommend(
        self,
        user_id: Any,
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> list[tuple[Any, float]]:
        """
        Get top-N item recommendations for a user.

        Args:
            user_id: Original user ID.
            n_items: Number of items to recommend.
            exclude_seen: If True, exclude items the user has already rated.

        Returns:
            list of (item_id, score) tuples sorted by score descending.

        Raises:
            ValueError: If model not trained or user unknown.
        """
        if not self.is_trained or self.model is None or self.train_df is None:
            raise ValueError("Model not trained. Call train() first.")

        if user_id not in self.user2idx:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.user2idx[user_id]

        # Get all item scores for this user
        item_scores: list[tuple[Any, float]] = []

        # Get seen items if excluding
        if exclude_seen:
            seen_items = set(
                self.train_df[self.train_df[self.user_col] == user_id][self.item_col]
            )
        else:
            seen_items = set()

        for item_id, item_idx in self.item2idx.items():
            if item_id in seen_items:
                continue
            score = self.model.score(user_idx, item_idx)
            item_scores.append((item_id, float(score)))

        # Sort by score descending and return top-N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_items]

    def save(self, checkpoint_dir: Path) -> None:
        """
        Save the trained model to a checkpoint directory.

        Args:
            checkpoint_dir: Directory to save model files.

        Raises:
            ValueError: If model not trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Cannot save untrained model.")

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model using Cornac's built-in save
        model_path = checkpoint_path / "bpr_model"
        self.model.save(str(model_path))

        # Save mappings and config
        config_path = checkpoint_path / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump({
                "user2idx": self.user2idx,
                "idx2user": self.idx2user,
                "item2idx": self.item2idx,
                "idx2item": self.idx2item,
                "train_df": self.train_df,
                "config": {
                    "k": self.k,
                    "max_iter": self.max_iter,
                    "learning_rate": self.learning_rate,
                    "lambda_reg": self.lambda_reg,
                    "seed": self.seed,
                }
            }, f)

        print(f"Model saved to {checkpoint_path}")

    def load(self, checkpoint_dir: Path) -> None:
        """
        Load a trained model from a checkpoint directory.

        Args:
            checkpoint_dir: Directory containing saved model files.

        Raises:
            FileNotFoundError: If checkpoint files not found.
        """
        from glob import glob

        try:
            import cornac  # noqa: F401
        except ImportError:
            raise ImportError("BPR requires cornac library.")

        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / "config.pkl"
        model_path = checkpoint_path / "bpr_model"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config and mappings
        with open(config_path, "rb") as f:
            data = pickle.load(f)

        self.user2idx = data["user2idx"]
        self.idx2user = data["idx2user"]
        self.item2idx = data["item2idx"]
        self.idx2item = data["idx2item"]
        self.train_df = data["train_df"]

        config = data["config"]
        self.k = config["k"]
        self.max_iter = config["max_iter"]
        self.learning_rate = config["learning_rate"]
        self.lambda_reg = config["lambda_reg"]
        self.seed = config["seed"]

        # Find the actual pkl file (Cornac saves to nested structure)
        # Pattern: bpr_model/BPR/timestamp.pkl
        pkl_files = glob(f"{model_path}/**/*.pkl", recursive=True)
        if not pkl_files:
            raise FileNotFoundError(f"No model pkl file found in {model_path}")

        # Use the most recent pkl file
        model_pkl = sorted(pkl_files)[-1]

        # Load BPR model directly from pkl file
        with open(model_pkl, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True

        print(f"Model loaded from {checkpoint_path}")

