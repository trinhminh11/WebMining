import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

def plot_loss(args):
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        history = json.load(f)

    # history is a list of dicts: [{'epoch': 0, 'loss': ...}, ...]
    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', label='EmerG Loss')
    plt.legend()
    plt.title('Training Loss per Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)

    output_path = args.output_path or json_path.parent / "loss_curve.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="app/backend/api/cold_start/checkpoints/loss_history.json")
    parser.add_argument('--output_path', type=str, default="", help="Optional output path")
    args = parser.parse_args()
    plot_loss(args)
