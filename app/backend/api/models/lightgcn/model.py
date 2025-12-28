import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import BaseRecommender


class LightGCNModule(nn.Module):
    """PyTorch module for LightGCN graph convolution."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_layers: int,
        latent_dim: int,
        adj_matrix: sp.spmatrix,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.latent_dim = latent_dim

        # Initialize embeddings
        self.E0 = nn.Embedding(n_users + n_items, latent_dim)
        nn.init.xavier_uniform_(self.E0.weight)

        # Normalize adjacency matrix
        self.norm_adj = self._normalize_adj(adj_matrix)

    def _normalize_adj(self, adj_matrix: sp.spmatrix) -> torch.sparse.FloatTensor:
        """Create normalized adjacency matrix for message passing."""
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = adj_matrix.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        # D^{-0.5} * A * D^{-0.5}
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        # Convert to sparse tensor
        coo = norm_adj.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate embeddings through GCN layers."""
        all_embeddings = [self.E0.weight]
        E_k = self.E0.weight

        for _ in range(self.n_layers):
            E_k = torch.sparse.mm(self.norm_adj, E_k)
            all_embeddings.append(E_k)

        # Mean of all layer embeddings
        all_embeddings = torch.stack(all_embeddings, dim=0)
        final_emb = torch.mean(all_embeddings, dim=0)

        user_emb = final_emb[: self.n_users]
        item_emb = final_emb[self.n_users :]

        return user_emb, item_emb

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass for BPR training."""
        user_emb, item_emb = self.propagate()

        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        u_emb0 = self.E0.weight[users]
        pos_emb0 = self.E0.weight[self.n_users + pos_items]
        neg_emb0 = self.E0.weight[self.n_users + neg_items]

        return u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0


class LightGCNRecommender(BaseRecommender):
    """
    LightGCN-based collaborative filtering recommender.

    Attributes:
        n_layers: Number of GCN propagation layers.
        latent_dim: Dimension of user/item embeddings.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for Adam optimizer.
        decay: L2 regularization weight.
        seed: Random seed.
    """

    def __init__(
        self,
        n_layers: int = 3,
        latent_dim: int = 64,
        n_epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 0.005,
        decay: float = 0.0001,
        seed: int = 42,
    ) -> None:
        super().__init__(name="lightgcn")
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.seed = seed

        self.model: LightGCNModule = None
        self.train_df: pd.DataFrame = None
        self.user_emb: torch.Tensor = None
        self.item_emb: torch.Tensor = None

        # ID mappings
        self.user2idx: dict[Any, int] = {}
        self.idx2user: dict[int, Any] = {}
        self.item2idx: dict[Any, int] = {}
        self.idx2item: dict[int, Any] = {}

        self.user_col = "userID"
        self.item_col = "itemID"
        self.rating_col = "rating"

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_adj_matrix(self, df: pd.DataFrame) -> sp.dok_matrix:
        """Build user-item adjacency matrix."""
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        R = sp.dok_matrix((n_users, n_items), dtype=np.float32)

        for _, row in df.iterrows():
            u = self.user2idx[row[self.user_col]]
            i = self.item2idx[row[self.item_col]]
            R[u, i] = 1.0

        return R

    def _sample_batch(
        self, df: pd.DataFrame, user_items: dict[int, list[int]], n_items: int
    ) -> tuple[list[int], list[int], list[int]]:
        """Sample a batch of (user, pos_item, neg_item) triplets."""
        users = random.sample(
            list(user_items.keys()), min(self.batch_size, len(user_items))
        )
        pos_items = []
        neg_items = []

        for u in users:
            pos = random.choice(user_items[u])
            pos_items.append(pos)

            neg = random.randint(0, n_items - 1)
            while neg in user_items[u]:
                neg = random.randint(0, n_items - 1)
            neg_items.append(neg)

        return users, pos_items, neg_items

    def _bpr_loss(
        self,
        u_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate BPR loss and regularization loss."""
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (
            (u_emb0.norm().pow(2) + pos_emb0.norm().pow(2) + neg_emb0.norm().pow(2))
            / 2
            / len(u_emb)
        )

        return mf_loss, reg_loss

    def train(self, train_data: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        """Train the LightGCN model."""
        from tqdm import tqdm

        self.train_df = train_data.copy()

        # Build mappings
        unique_users = train_data[self.user_col].unique()
        unique_items = train_data[self.item_col].unique()

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        # Build interaction matrix
        adj_matrix = self._build_adj_matrix(train_data)

        # User items dict for sampling
        user_items: dict[int, list[int]] = {}
        for _, row in train_data.iterrows():
            u = self.user2idx[row[self.user_col]]
            i = self.item2idx[row[self.item_col]]
            if u not in user_items:
                user_items[u] = []
            user_items[u].append(i)

        # Initialize model
        self.model = LightGCNModule(
            n_users, n_items, self.n_layers, self.latent_dim, adj_matrix
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with loss tracking
        n_batches = max(1, len(train_data) // self.batch_size)
        loss_history: list[float] = []

        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(n_batches):
                optimizer.zero_grad()

                users, pos_items, neg_items = self._sample_batch(
                    train_data, user_items, n_items
                )

                u_t = torch.LongTensor(users)
                pos_t = torch.LongTensor(pos_items)
                neg_t = torch.LongTensor(neg_items)

                u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0 = self.model(
                    u_t, pos_t, neg_t
                )

                mf_loss, reg_loss = self._bpr_loss(
                    u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0
                )
                loss = mf_loss + self.decay * reg_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

        # Get final embeddings
        self.model.eval()
        with torch.no_grad():
            self.user_emb, self.item_emb = self.model.propagate()

        self.is_trained = True

        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_layers": self.n_layers,
            "loss_history": loss_history,
        }

    def predict(self, user_id: Any, item_id: Any) -> float:
        """Predict score for user-item pair."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        if user_id not in self.user2idx:
            raise ValueError(f"Unknown user: {user_id}")
        if item_id not in self.item2idx:
            raise ValueError(f"Unknown item: {item_id}")

        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]

        score = torch.dot(self.user_emb[u_idx], self.item_emb[i_idx]).item()
        return float(score)

    def recommend(
        self, user_id: Any, n_items: int = 10, exclude_seen: bool = True
    ) -> list[tuple[Any, float]]:
        """Get top-N recommendations for a user."""
        if not self.is_trained or self.train_df is None:
            raise ValueError("Model not trained.")

        if user_id not in self.user2idx:
            raise ValueError(f"Unknown user: {user_id}")

        u_idx = self.user2idx[user_id]

        # Get scores for all items
        scores = torch.matmul(self.user_emb[u_idx], self.item_emb.T)

        # Exclude seen items
        if exclude_seen:
            seen = self.train_df[self.train_df[self.user_col] == user_id][self.item_col]
            seen_idx = [self.item2idx[i] for i in seen if i in self.item2idx]
            scores[seen_idx] = float("-inf")

        # Get top items
        top_idx = torch.argsort(scores, descending=True)[:n_items]

        return [(self.idx2item[i.item()], scores[i].item()) for i in top_idx]

    def save(self, checkpoint_dir: Path) -> None:
        """Save model to checkpoint directory."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        torch.save(self.user_emb, path / "user_emb.pt")
        torch.save(self.item_emb, path / "item_emb.pt")

        # Save config
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(
                {
                    "user2idx": self.user2idx,
                    "idx2user": self.idx2user,
                    "item2idx": self.item2idx,
                    "idx2item": self.idx2item,
                    "train_df": self.train_df,
                    "config": {
                        "n_layers": self.n_layers,
                        "latent_dim": self.latent_dim,
                        "n_epochs": self.n_epochs,
                    },
                },
                f,
            )

        print(f"Model saved to {path}")

    def load(self, checkpoint_dir: Path) -> None:
        """Load model from checkpoint directory."""
        path = Path(checkpoint_dir)

        if not (path / "config.pkl").exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path / "config.pkl", "rb") as f:
            data = pickle.load(f)

        self.user2idx = data["user2idx"]
        self.idx2user = data["idx2user"]
        self.item2idx = data["item2idx"]
        self.idx2item = data["idx2item"]
        self.train_df = data["train_df"]

        config = data["config"]
        self.n_layers = config["n_layers"]
        self.latent_dim = config["latent_dim"]
        self.n_epochs = config["n_epochs"]

        self.user_emb = torch.load(path / "user_emb.pt")
        self.item_emb = torch.load(path / "item_emb.pt")

        self.is_trained = True
        print(f"Model loaded from {path}")
