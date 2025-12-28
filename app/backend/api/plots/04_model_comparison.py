# Cell 4: Model Performance Comparison Bar Chart
# This cell creates a bar chart comparing Precision@10 and Recall@10 across models

# Prepare data
comparison_data = []
for model_name, metrics in all_metrics.items():
    comparison_data.append({
        "Model": model_name.upper(),
        "Precision@10": metrics.get("final_precision", metrics.get("eval_metrics", {}).get("precision@10", 0)),
        "Recall@10": metrics.get("final_recall", metrics.get("eval_metrics", {}).get("recall@10", 0)),
        "Train Time (s)": metrics.get("train_time", metrics.get("train_history", {}).get("train_time_seconds", 0)),
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values("Precision@10", ascending=False)

print(comparison_df.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision
ax1 = axes[0]
bars1 = ax1.bar(comparison_df["Model"], comparison_df["Precision@10"], color=sns.color_palette("husl", len(comparison_df)))
ax1.set_xlabel("Model", fontsize=12)
ax1.set_ylabel("Precision@10", fontsize=12)
ax1.set_title("Precision@10 by Model", fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar, val in zip(bars1, comparison_df["Precision@10"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Recall
ax2 = axes[1]
bars2 = ax2.bar(comparison_df["Model"], comparison_df["Recall@10"], color=sns.color_palette("husl", len(comparison_df)))
ax2.set_xlabel("Model", fontsize=12)
ax2.set_ylabel("Recall@10", fontsize=12)
ax2.set_title("Recall@10 by Model", fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, val in zip(bars2, comparison_df["Recall@10"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(CHECKPOINTS_DIR / "model_comparison_metrics.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved to {CHECKPOINTS_DIR / 'model_comparison_metrics.png'}")
