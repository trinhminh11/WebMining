# Cell 2: Load all model metrics
# This cell loads metrics.json from each model checkpoint

def load_model_metrics(model_name: str) -> dict:
    """Load metrics from a model's checkpoint directory."""
    metrics_path = CHECKPOINTS_DIR / model_name / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# Load all metrics
models = ["mf", "bpr", "bivae", "lightgcn", "lightfm"]
all_metrics = {}

for model in models:
    metrics = load_model_metrics(model)
    if metrics:
        all_metrics[model] = metrics
        print(f"Loaded {model}: {list(metrics.keys())}")
    else:
        print(f"No metrics found for {model}")

print(f"\nLoaded {len(all_metrics)} models")
