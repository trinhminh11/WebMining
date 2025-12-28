"""
Abstract Base Model for Recommendation Systems.

This module defines the abstract base class that all recommendation models
must implement. It provides a consistent interface for training, prediction,
recommendation, and model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json


class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.

    All recommendation models should inherit from this class and implement
    the required abstract methods for training, prediction, and persistence.

    Attributes:
        name: Unique identifier for the model type.
        model: The underlying model object (set after training).
        is_trained: Flag indicating whether the model has been trained.

    Example:
        >>> class MyRecommender(BaseRecommender):
        ...     def train(self, train_data, **kwargs):
        ...         # Implementation
        ...         pass
        ...     # ... implement other abstract methods
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the base recommender.

        Args:
            name: Unique identifier for the model type (e.g., 'mf', 'ncf').
        """
        self.name: str = name
        self.model = None
        self.is_trained: bool = False

    @abstractmethod
    def train(self, train_data: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Train the recommendation model.

        Args:
            train_data: Training data. Format depends on the specific model
                implementation (typically a pandas DataFrame).
            **kwargs: Additional model-specific training arguments.

        Returns:
            Dictionary containing training statistics and history.

        Raises:
            ValueError: If training data is invalid or incompatible.
        """
        pass

    @abstractmethod
    def predict(self, user_id: Any, item_id: Any) -> float:
        """
        Predict the rating/score for a user-item pair.

        Args:
            user_id: Unique identifier for the user.
            item_id: Unique identifier for the item.

        Returns:
            Predicted rating or score for the user-item pair.

        Raises:
            ValueError: If the model has not been trained.
            KeyError: If user_id or item_id is unknown.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: Any,
        n_items: int = 10,
        exclude_seen: bool = True
    ) -> list[tuple[Any, float]]:
        """
        Get top-N item recommendations for a user.

        Args:
            user_id: Unique identifier for the user.
            n_items: Number of items to recommend.
            exclude_seen: If True, exclude items the user has already interacted with.

        Returns:
            list of (item_id, predicted_score) tuples, sorted by score descending.

        Raises:
            ValueError: If the model has not been trained.
            KeyError: If user_id is unknown.
        """
        pass

    @abstractmethod
    def save(self, checkpoint_dir: Path) -> None:
        """
        Save the trained model to a checkpoint directory.

        Args:
            checkpoint_dir: Directory path where model files will be saved.
                The directory will be created if it doesn't exist.

        Raises:
            ValueError: If the model has not been trained.
        """
        pass

    @abstractmethod
    def load(self, checkpoint_dir: Path) -> None:
        """
        Load a trained model from a checkpoint directory.

        Args:
            checkpoint_dir: Directory path containing saved model files.

        Raises:
            FileNotFoundError: If the checkpoint directory or required files don't exist.
            ValueError: If the checkpoint files are corrupted or incompatible.
        """
        pass

    def save_metrics(self, checkpoint_dir: Path, metrics: dict[str, Any]) -> None:
        """
        Save evaluation metrics to a JSON file in the checkpoint directory.

        Args:
            checkpoint_dir: Directory where metrics.json will be saved.
            metrics: Dictionary containing training history and evaluation metrics.
                Expected structure:
                {
                    "train_history": {...},
                    "eval_metrics": {"precision@k": float, "recall@k": float, ...}
                }
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

    def load_metrics(self, checkpoint_dir: Path) -> dict[str, Any] | None:
        """
        Load evaluation metrics from the checkpoint directory.

        Args:
            checkpoint_dir: Directory containing metrics.json.

        Returns:
            Dictionary with metrics if found, None otherwise.
        """
        metrics_path = Path(checkpoint_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
