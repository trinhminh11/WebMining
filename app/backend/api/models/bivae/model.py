import pickle
import sys
from glob import glob
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import BaseRecommender


class BiVAERecommender(BaseRecommender):
    """
    BiVAE-based recommender using the Cornac library.

    This model uses Bilateral Variational Autoencoders to learn
    user and item latent representations for collaborative filtering.

    Attributes:
        k: Latent dimension for embeddings.
        encoder_structure: list of hidden layer sizes for encoders.
        act_fn: Activation function ('tanh', 'relu', etc.).
        likelihood: Likelihood function ('pois', 'bern', 'gaus').
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimizer.
        use_gpu: Whether to use GPU if available.
        verbose: Whether to print training progress.
        seed: Random seed for reproducibility.
        model: The trained Cornac BiVAECF model.
        train_set: The Cornac Dataset object.
        user2idx: Mapping from original user IDs to Cornac indices.
        item2idx: Mapping from original item IDs to Cornac indices.

    Example:
        >>> model = BiVAERecommender(k=50, n_epochs=100)
        >>> stats = model.train(train_df)
        >>> recs = model.recommend(user_id=123, n_items=10)
    """

    def __init__(
        self,
        k: int = 50,
        encoder_structure: Optional[list[int]] = None,
        act_fn: str = "tanh",
        likelihood: str = "pois",
        n_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        use_gpu: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the BiVAE recommender.

        Args:
            k: Latent dimension for user/item embeddings.
            encoder_structure: list of hidden layer sizes.
                Defaults to [100] if None.
            act_fn: Activation function ('tanh', 'relu', 'sigmoid').
            likelihood: Likelihood for reconstruction:
                - 'pois': Poisson (for count data)
                - 'bern': Bernoulli (for binary data)
                - 'gaus': Gaussian
            n_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            use_gpu: Use GPU if available.
            verbose: Print training progress.
            seed: Random seed.
        """
        super().__init__(name="bivae")
        self.k: int = k
        self.encoder_structure: list[int] = encoder_structure or [100]
        self.act_fn: str = act_fn
        self.likelihood: str = likelihood
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.use_gpu: bool = use_gpu
        self.verbose: bool = verbose
        self.seed: int = seed

        self.model = None
        self.train_set = None
        self.train_df: pd.DataFrame = None

        # ID mappings
        self.user2idx: dict[Any, int] = {}
        self.idx2user: dict[int, Any] = {}
        self.item2idx: dict[Any, int] = {}
        self.idx2item: dict[int, Any] = {}

        # Column names
        self.user_col: str = "userID"
        self.item_col: str = "itemID"
        self.rating_col: str = "rating"

    def train(self, train_data: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        """
        Train the BiVAE model.

        Args:
            train_data: DataFrame with userID, itemID, rating columns.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing training statistics.
        """
        try:
            import cornac
            import torch
        except ImportError as e:
            raise ImportError(f"BiVAE requires cornac and torch. Error: {e}")

        self.train_df = train_data.copy()

        # Create Cornac dataset
        uir_tuples = list(
            train_data[[self.user_col, self.item_col, self.rating_col]].itertuples(
                index=False, name=None
            )
        )

        self.train_set = cornac.data.Dataset.from_uir(uir_tuples, seed=self.seed)

        # Store ID mappings
        self.user2idx = self.train_set.uid_map
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.item2idx = self.train_set.iid_map
        self.idx2item = {v: k for k, v in self.item2idx.items()}

        # Check GPU availability
        use_gpu = self.use_gpu and torch.cuda.is_available()

        # Initialize BiVAE model
        self.model = cornac.models.BiVAECF(
            k=self.k,
            encoder_structure=self.encoder_structure,
            act_fn=self.act_fn,
            likelihood=self.likelihood,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            use_gpu=use_gpu,
            verbose=self.verbose,
        )

        # Train
        self.model.fit(self.train_set)
        self.is_trained = True

        return {
            "n_users": self.train_set.num_users,
            "n_items": self.train_set.num_items,
            "k": self.k,
            "n_epochs": self.n_epochs,
        }

    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict the preference score for a user-item pair."""
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
        self, user_id: Any, n_items: int = 10, exclude_seen: bool = True
    ) -> list[tuple[Any, float]]:
        """Get top-N item recommendations for a user."""
        if not self.is_trained or self.model is None or self.train_df is None:
            raise ValueError("Model not trained. Call train() first.")

        if user_id not in self.user2idx:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_idx = self.user2idx[user_id]

        # Get seen items
        if exclude_seen:
            seen_items = set(
                self.train_df[self.train_df[self.user_col] == user_id][self.item_col]
            )
        else:
            seen_items = set()

        # Score all items
        item_scores: list[tuple[Any, float]] = []
        for item_id, item_idx in self.item2idx.items():
            if item_id in seen_items:
                continue
            score = self.model.score(user_idx, item_idx)
            item_scores.append((item_id, float(score)))

        # Sort and return top-N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_items]

    def save(self, checkpoint_dir: Path) -> None:
        """Save the trained model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained.")

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_path / "bivae_model"
        self.model.save(str(model_path))

        # Save config
        config_path = checkpoint_path / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(
                {
                    "user2idx": self.user2idx,
                    "idx2user": self.idx2user,
                    "item2idx": self.item2idx,
                    "idx2item": self.idx2item,
                    "train_df": self.train_df,
                    "config": {
                        "k": self.k,
                        "encoder_structure": self.encoder_structure,
                        "act_fn": self.act_fn,
                        "likelihood": self.likelihood,
                        "n_epochs": self.n_epochs,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                        "seed": self.seed,
                    },
                },
                f,
            )

        print(f"Model saved to {checkpoint_path}")

    def load(self, checkpoint_dir: Path) -> None:
        """Load a trained model."""
        checkpoint_path = Path(checkpoint_dir)
        config_path = checkpoint_path / "config.pkl"
        model_path = checkpoint_path / "bivae_model"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load config
        with open(config_path, "rb") as f:
            data = pickle.load(f)

        self.user2idx = data["user2idx"]
        self.idx2user = data["idx2user"]
        self.item2idx = data["item2idx"]
        self.idx2item = data["idx2item"]
        self.train_df = data["train_df"]

        config = data["config"]
        self.k = config["k"]
        self.encoder_structure = config["encoder_structure"]
        self.act_fn = config["act_fn"]
        self.likelihood = config["likelihood"]
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.seed = config["seed"]

        # Find and load pkl file
        pkl_files = glob(f"{model_path}/**/*.pkl", recursive=True)
        if not pkl_files:
            raise FileNotFoundError(f"No model pkl found in {model_path}")

        model_pkl = sorted(pkl_files)[-1]
        with open(model_pkl, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True
        print(f"Model loaded from {checkpoint_path}")
