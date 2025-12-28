# Cell 5: Training Time Comparison
# This cell compares training time across models

fig, ax = plt.subplots(figsize=(10, 5))

colors = sns.color_palette("husl", len(comparison_df))
bars = ax.barh(comparison_df["Model"], comparison_df["Train Time (s)"], color=colors)

ax.set_xlabel("Training Time (seconds)", fontsize=12)
ax.set_ylabel("Model", fontsize=12)
ax.set_title("Training Time by Model", fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, comparison_df["Train Time (s)"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}s', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(CHECKPOINTS_DIR / "training_time_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved to {CHECKPOINTS_DIR / 'training_time_comparison.png'}")
