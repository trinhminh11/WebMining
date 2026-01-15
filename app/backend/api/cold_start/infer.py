import torch
import json
import argparse
from pathlib import Path
import numpy as np
from app.backend.api.cold_start.dataset import BooksColdStartDataLoader
from app.backend.api.cold_start.model import GNN, EmerG

def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loader (Test Split)
    data_dir = args.data_dir
    loader = BooksColdStartDataLoader(data_dir, device=device, bsz=args.batch_size, shuffle=False)
    test_dl = loader.get_loader('test')

    # Model
    print("Loading Model...")
    embed_dim = args.embed_dim
    description = loader.description
    gnn = GNN(description, embed_dim, gnn_layers=args.gnn_layers, device=device)
    item_feat_dim = embed_dim * 4
    model = EmerG(gnn, item_feat_dim, device=device).to(device)

    # Load Checkpoint
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = {}
    total_loss = 0.0
    steps = 0

    print("Starting Inference...")
    with torch.no_grad():
        for batch_features, batch_labels in test_dl:
            # Group by Item for MAML-style evaluation
            items = batch_features['item_id']
            unique_items = torch.unique(items)

            for item_idx in unique_items:
                mask = (items == item_idx)
                item_indices = torch.nonzero(mask, as_tuple=True)[0]

                # In test time (Cold Start):
                # We assume we have a few support samples (warm-up) and we predict on the rest.
                # If these are purely COLD items, we might have 0 samples initially?
                # EmerG scenario: We use the support set to adapt.
                # If 'test' split contains cold items, the model has literally never seen this ID embedding.
                # The ID embedding is random/zero.
                # The GraphGenerator uses ITEM FEATURES (Author, Year, Publisher) to generate a graph.
                # This graph + features drive the prediction.

                # If we have interactions in this batch, we can treat some as support to refine ID?
                # Or just Zero-Shot (Support=Empty)?
                # EmerG paper usually does few-shot cold-start.

                if len(item_indices) < 2:
                    continue

                # Let's use first 50% as support to simulate "few-shot" adaptation if available
                k_support = max(1, len(item_indices) // 2)
                support_idx = item_indices[:k_support]
                query_idx = item_indices[k_support:]

                if len(query_idx) == 0: continue

                support_x = {k: v[support_idx] for k, v in batch_features.items()}
                support_y = batch_labels[support_idx].to(device)

                query_x = {k: v[query_idx] for k, v in batch_features.items()}
                query_y = batch_labels[query_idx].to(device)

                pred, loss = model(support_x, support_y, query_x)

                # Store results
                # Map back to real IDs if possible, but for now store by Item_Idx
                iid = int(item_idx.item())
                if iid not in results:
                    results[iid] = {'preds': [], 'labels': []}

                results[iid]['preds'].extend(pred.cpu().numpy().flatten().tolist())
                results[iid]['labels'].extend(query_y.cpu().numpy().flatten().tolist())

                total_loss += loss.item()
                steps += 1

    avg_loss = total_loss / steps if steps > 0 else 0.0
    print(f"Inference Complete. Avg Loss: {avg_loss:.4f}")

    # Save Results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            'avg_loss': avg_loss,
            'details': results
        }, f)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="Books/app/backend/api/data")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="Books/app/backend/api/cold_start/results.json")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    infer(args)
