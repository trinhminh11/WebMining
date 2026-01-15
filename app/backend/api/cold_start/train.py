import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import os
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np

from app.backend.api.cold_start.dataset import BooksColdStartDataLoader
from app.backend.api.cold_start.model import GNN, EmerG

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loader
    data_dir = args.data_dir
    loader = BooksColdStartDataLoader(data_dir, device=device, bsz=args.batch_size, shuffle=True)
    train_dl = loader.get_loader('train')

    # Model
    print("Initializing Model...")
    embed_dim = args.embed_dim

    # Description
    description = loader.description

    # GNN (Base Model)
    gnn = GNN(description, embed_dim, gnn_layers=args.gnn_layers, device=device)

    # calculate item_features_dim for GraphGenerator
    # Item ID (embed_dim) + Author (embed_dim) + Year (embed_dim) + Publisher (embed_dim)
    # Total = 4 * embed_dim
    item_feat_dim = embed_dim * 4

    model = EmerG(gnn, item_feat_dim, device=device, inner_lr=args.inner_lr).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.outer_lr)

    # Checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []

    print("Starting Training...")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0

        for batch_features, batch_labels in train_dl:
            # MAML Batch: We treat each batch as a task or set of tasks?
            # Standard MAML: Sample task -> Support Set, Query Set.
            # In Cold Start Item Rec:
            # Task = Recommend for Item i.
            # Support = Known users rating Item i.
            # Query = Other users rating Item i (to predict).

            # Our DataLoader currently returns mixed batches of (user, item) interactions.
            # This is NOT ideal for MAML meta-learning per-item.
            # However, adapting existing dataloader:

            # For strict cold-start simulation:
            # We need to group by Item_Idx.
            # Since simpler loader was requested/implemented, we will simulate tasks by:
            # 1. Group batch by Item_Idx.
            # 2. For each unique item in batch with > K samples:
            #    Split into Support and Query.
            #    Run Inner Loop.
            #    Accumulate Query Loss.

            # Grouping
            items = batch_features['item_id']
            unique_items = torch.unique(items)

            outer_loss = 0.0
            valid_tasks = 0

            for item_idx in unique_items:
                mask = (items == item_idx)
                # total samples for this item in batch
                item_indices = torch.nonzero(mask, as_tuple=True)[0]

                if len(item_indices) < 4: # Need at least some samples for support/query
                    continue

                # Split Support/Query (e.g., 50/50 or fixed support size)
                # Let's say k_support = half or min(validation split)
                k_support = len(item_indices) // 2
                support_idx = item_indices[:k_support]
                query_idx = item_indices[k_support:]

                # Construct sub-batches
                support_x = {k: v[support_idx] for k, v in batch_features.items()}
                support_y = batch_labels[support_idx].to(device)

                query_x = {k: v[query_idx] for k, v in batch_features.items()}
                query_y = batch_labels[query_idx].to(device)

                # Forward (Meta-Forward handles inner loop/graph generation)
                pred_query, loss_query = model(support_x, support_y, query_x)

                # Loss on Query Set
                task_loss = model.criterion(pred_query, query_y.view(-1, 1))
                outer_loss += task_loss
                valid_tasks += 1

            if valid_tasks > 0:
                outer_loss = outer_loss / valid_tasks

                optimizer.zero_grad()
                outer_loss.backward()
                optimizer.step()

                epoch_loss += outer_loss.item()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0
        loss_history.append({'epoch': epoch, 'loss': avg_loss})
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), ckpt_dir / f"model_epoch_{epoch+1}.pt")

    # Save Loss
    with open(ckpt_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="Books/app/backend/api/data")
    parser.add_argument('--ckpt_dir', type=str, default="Books/app/backend/api/cold_start/checkpoints")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256) # Larger batch to ensure multiple samples per item
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--outer_lr', type=float, default=0.0001)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    train(args)
