# Cell 6: Summary Table
# This cell creates and displays a summary table of all results

# Load comparison CSV
comparison_csv = CHECKPOINTS_DIR / "model_comparison.csv"
if comparison_csv.exists():
    results_df = pd.read_csv(comparison_csv)
    results_df = results_df.sort_values("precision@10", ascending=False)
    results_df["Rank"] = range(1, len(results_df) + 1)

    print("=" * 60)
    print("MODEL RANKING (by Precision@10)")
    print("=" * 60)
    print(results_df[["Rank", "model", "precision@10", "recall@10"]].to_string(index=False))
else:
    print("No comparison CSV found. Run training first.")

# Create styled table for display
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')

table_data = results_df[["Rank", "model", "precision@10", "recall@10"]].values.tolist()
headers = ["Rank", "Model", "Precision@10", "Recall@10"]

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colColours=['#4CAF50'] * 4,
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')

plt.title("Model Performance Ranking", fontsize=14, fontweight='bold', pad=20)
plt.savefig(CHECKPOINTS_DIR / "model_ranking_table.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved to {CHECKPOINTS_DIR / 'model_ranking_table.png'}")
