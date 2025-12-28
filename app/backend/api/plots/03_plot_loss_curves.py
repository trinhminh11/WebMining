# Cell 3: Plot Learning Curves (Training Loss)
# This cell plots training loss per epoch for models that track it

fig, ax = plt.subplots(figsize=(10, 6))

for model_name, metrics in all_metrics.items():
    if "learning_curve" in metrics:
        curve = metrics["learning_curve"]
        epochs = curve.get("epochs", [])

        # Try different loss keys
        loss = curve.get("train_loss", curve.get("train_rmse", []))

        if epochs and loss:
            ax.plot(epochs, loss, marker='o', markersize=4, label=model_name.upper(), linewidth=2)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.set_title("Training Loss per Epoch", fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CHECKPOINTS_DIR / "learning_curves_loss.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved to {CHECKPOINTS_DIR / 'learning_curves_loss.png'}")
