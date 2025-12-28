import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import surprise
from surprise import Dataset, Reader

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import BaseRecommender


class SVDRecommender(BaseRecommender):
    """
    SVD-based Matrix Factorization recommender using Surprise library.

    This model uses Singular Value Decomposition to learn latent factors
    for users and items, enabling rating prediction and item recommendation.

    Attributes:
        n_factors: Number of latent factors.
        n_epochs: Number of training epochs.
        lr_all: Learning rate for all parameters.
        reg_all: Regularization term for all parameters.
        random_state: Random seed for reproducibility.
        verbose: Whether to print training progress.
        model: The trained Surprise SVD model.
        trainset: The Surprise Trainset object.
        train_df: Original training DataFrame.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        rating_col: Name of the rating column.

    Example:
        >>> model = SVDRecommender(n_factors=100, n_epochs=20)
        >>> stats = model.train(train_df, rating_scale=(0, 10))
        >>> print(f"Trained on {stats['n_users']} users")
        >>> score = model.predict(user_id=123, item_id=456)
        >>> recs = model.recommend(user_id=123, n_items=10)
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the SVD recommender.

        Args:
            n_factors: Number of latent factors for user/item embeddings.
                Higher values capture more complex patterns but may overfit.
            n_epochs: Number of training epochs (passes over the data).
            lr_all: Learning rate for all parameters (biases and factors).
            reg_all: L2 regularization term for all parameters.
            random_state: Random seed for reproducibility.
            verbose: If True, print training progress for each epoch.
        """
        super().__init__(name="mf")
        self.n_factors: int = n_factors
        self.n_epochs: int = n_epochs
        self.lr_all: float = lr_all
        self.reg_all: float = reg_all
        self.random_state: int = random_state
        self.verbose: bool = verbose

        self.model: surprise.SVD = None
        self.trainset: surprise.Trainset = None
        self.train_df: pd.DataFrame = None

        # ID mappings (for compatibility with other models)
        self.user2idx: dict[Any, int] = {}
        self.idx2user: dict[int, Any] = {}
        self.item2idx: dict[Any, int] = {}
        self.idx2item: dict[int, Any] = {}

        # Column names (matching books.py defaults)
        self.user_col: str = "userID"
        self.item_col: str = "itemID"
        self.rating_col: str = "rating"

    def train(
        self,
        train_data: pd.DataFrame,
        rating_scale: tuple[float, float] = (0, 10),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the SVD model on the provided data.

        Args:
            train_data: DataFrame containing user-item-rating triplets.
                Must have columns matching user_col, item_col, and rating_col.
            rating_scale: tuple of (min_rating, max_rating) for the rating scale.
            **kwargs: Additional arguments (unused, for interface compatibility).

        Returns:
            Dictionary containing training statistics:
                - n_users: Number of unique users in training set
                - n_items: Number of unique items in training set
                - n_ratings: Total number of ratings

        Raises:
            ValueError: If train_data is missing required columns.
        """
        required_cols = [self.user_col, self.item_col, self.rating_col]
        missing_cols = [col for col in required_cols if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.train_df = train_data.copy()

        # Create Surprise Dataset
        reader = Reader(rating_scale=rating_scale)
        data = Dataset.load_from_df(
            train_data[[self.user_col, self.item_col, self.rating_col]], reader=reader
        )
        self.trainset = data.build_full_trainset()

        # Build ID mappings for compatibility with other models
        unique_users = train_data[self.user_col].unique()
        unique_items = train_data[self.item_col].unique()
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        # Initialize and train SVD
        self.model = surprise.SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.model.fit(self.trainset)
        self.is_trained = True

        return {
            "n_users": self.trainset.n_users,
            "n_items": self.trainset.n_items,
            "n_ratings": self.trainset.n_ratings,
        }

    def predict(self, user_id: Any, item_id: Any) -> float:
        """
        Predict the rating for a user-item pair.

        Args:
            user_id: Unique identifier for the user.
            item_id: Unique identifier for the item (ISBN for books).

        Returns:
            Predicted rating value.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        prediction = self.model.predict(user_id, item_id)
        return float(prediction.est)

    def predict_batch(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict ratings for multiple user-item pairs.

        Args:
            test_data: DataFrame with user_col and item_col columns.

        Returns:
            DataFrame with original columns plus 'prediction' column.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions: list[float] = []
        for _, row in test_data.iterrows():
            pred = self.model.predict(row[self.user_col], row[self.item_col])
            predictions.append(pred.est)

        result = test_data.copy()
        result["prediction"] = predictions
        return result

    def recommend(
        self, user_id: Any, n_items: int = 10, exclude_seen: bool = True
    ) -> list[tuple[Any, float]]:
        """
        Get top-N item recommendations for a user.

        Args:
            user_id: Unique identifier for the user.
            n_items: Number of items to recommend.
            exclude_seen: If True, exclude items the user has already rated.

        Returns:
            list of (item_id, predicted_score) tuples, sorted by score descending.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_trained or self.model is None or self.train_df is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get all items
        all_items = set(self.train_df[self.item_col].unique())

        # Get items the user has already rated
        if exclude_seen:
            seen_items = set(
                self.train_df[self.train_df[self.user_col] == user_id][self.item_col]
            )
            candidate_items = all_items - seen_items
        else:
            candidate_items = all_items

        # Get predictions for all candidate items
        predictions: list[tuple[Any, float]] = []
        for item_id in candidate_items:
            pred = self.model.predict(user_id, item_id)
            predictions.append((item_id, float(pred.est)))

        # Sort by predicted score and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_items]

    def compute_ranking_predictions(
        self,
        train_data: pd.DataFrame,
        remove_seen: bool = True,
        n_sample_users: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute ranking predictions for all users and items.

        This is useful for evaluation metrics that require predictions
        for all user-item pairs.

        Args:
            train_data: Training data to determine seen items.
            remove_seen: If True, exclude items users have already rated.
            n_sample_users: If provided, sample this many users for efficiency.

        Returns:
            DataFrame with userID, itemID, prediction columns.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        import numpy as np

        all_users = train_data[self.user_col].unique()
        all_items = train_data[self.item_col].unique()

        # Sample users if specified
        if n_sample_users is not None and len(all_users) > n_sample_users:
            np.random.seed(42)
            all_users = np.random.choice(all_users, n_sample_users, replace=False)

        results: list[dict[str, Any]] = []
        for user_id in all_users:
            seen_items = set(
                train_data[train_data[self.user_col] == user_id][self.item_col]
            )

            for item_id in all_items:
                if remove_seen and item_id in seen_items:
                    continue
                pred = self.model.predict(user_id, item_id)
                results.append(
                    {
                        self.user_col: user_id,
                        self.item_col: item_id,
                        "prediction": pred.est,
                    }
                )

        return pd.DataFrame(results)

    def save(self, checkpoint_dir: Path) -> None:
        """
        Save the trained model to a checkpoint directory.

        Saves the model, trainset, training dataframe, and configuration
        to a pickle file.

        Args:
            checkpoint_dir: Directory path where model.pkl will be saved.

        Raises:
            ValueError: If the model has not been trained.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "trainset": self.trainset,
                    "train_df": self.train_df,
                    "config": {
                        "n_factors": self.n_factors,
                        "n_epochs": self.n_epochs,
                        "lr_all": self.lr_all,
                        "reg_all": self.reg_all,
                        "random_state": self.random_state,
                    },
                },
                f,
            )

        print(f"Model saved to {model_path}")

    def load(self, checkpoint_dir: Path) -> None:
        """
        Load a trained model from a checkpoint directory.

        Args:
            checkpoint_dir: Directory path containing model.pkl.

        Raises:
            FileNotFoundError: If model.pkl doesn't exist in the directory.
        """
        model_path = Path(checkpoint_dir) / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.trainset = data["trainset"]
        self.train_df = data["train_df"]

        config: dict[str, Any] = data.get("config", {})
        self.n_factors = config.get("n_factors", self.n_factors)
        self.n_epochs = config.get("n_epochs", self.n_epochs)
        self.lr_all = config.get("lr_all", self.lr_all)
        self.reg_all = config.get("reg_all", self.reg_all)
        self.random_state = config.get("random_state", self.random_state)

        self.is_trained = True
        print(f"Model loaded from {model_path}")
