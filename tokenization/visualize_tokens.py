import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Config
INPUT_CSV = "PATH/Thesis_deliverables_code/tokenization/outputs/llm_ranking_gemma_tokenized.csv"
OUTPUT_SCATTER = "PATH/Thesis_deliverables_code/tokenization/outputs/token_counts_scatter.pdf"
OUTPUT_BARCHART = "PATH/Thesis_deliverables_code/tokenization/outputs/token_counts_barchart.pdf"

# The fice tokenizer we use to plot and see which category gets the most tokens
TOKENIZERS = {
    "BERT":  "bert_token_count",
    "LLaMA 4": "llama4_token_count",
    "Gemma 4": "gemma4_token_count",
    "OpenAI":  "openai_token_count",
    "Mistral": "mistral_token_count",
}

CAT_COLORS = {
    1: "#1b9e77",  # dark teal
    2: "#d95f02",  # orange
    3: "#7570b3",  # purple
    4: "#e7298a",  # pink
    5: "#66a61e",  # green
}

CAT_LABELS = {
    1: "Cat 1 (≥80)",
    2: "Cat 2 (60–79)",
    3: "Cat 3 (40–59)",
    4: "Cat 4 (20–39)",
    5: "Cat 5 (<20)",
}

# Load data
df = pl.read_csv(INPUT_CSV)
positions = np.arange(1, len(df) + 1)
categories = df["category"].to_numpy()

# Draw vertical lines where the category change to visualize in the scatter plot
boundaries = []
for i in range(1, len(categories)):
    if categories[i] != categories[i - 1]:
        boundaries.append((i + 0.5, categories[i - 1], categories[i]))

# One scatter panel per tokenizer, x-axis rank position, y-axis is how many tokens that term got
fig, axes = plt.subplots(3, 2, figsize=(6.5, 8), constrained_layout=True)
axes_flat = axes.flatten()

for idx, (name, col) in enumerate(TOKENIZERS.items()):
    ax = axes_flat[idx]
    token_counts = df[col].to_numpy()

    for cat in sorted(CAT_COLORS):
        mask = categories == cat
        ax.scatter(
            positions[mask], token_counts[mask],
            alpha=0.4, s=12, color=CAT_COLORS[cat], label=CAT_LABELS[cat],
        )

    for bx, cat_before, cat_after in boundaries:
        ax.axvline(bx, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Token count")
    ax.yaxis.get_major_locator().set_params(integer=True)

    if idx >= 3:
        ax.set_xlabel("LLM-based domain-specificity category (1 = highest score)")

# hide panel six becuase we only have 5 tokenizers
axes_flat[5].set_visible(False)

# Shared legend from first subplot
handles, labels = axes_flat[0].get_legend_handles_labels()

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Token count by LLM-based rank position across tokenizers",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
fig.savefig(OUTPUT_SCATTER, bbox_inches="tight")

# Second figure showing the average token count per category in a barchart
fig_bar, ax_bar = plt.subplots(figsize=(6.5, 5))

cat_vals = sorted(CAT_COLORS.keys())
x = np.arange(len(cat_vals))
bar_width = 0.15

all_means = {}
for i, (name, col) in enumerate(TOKENIZERS.items()):
    means = []
    for cat in cat_vals:
        subset = df.filter(pl.col("category") == cat)[col]
        means.append(subset.mean() if len(subset) > 0 else 0)
    all_means[name] = means
    bars = ax_bar.bar(x + i * bar_width, means, bar_width, label=name, alpha=0.85)
    for bar, val in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

ax_bar.set_xticks(x + bar_width * (len(TOKENIZERS) - 1) / 2)
ax_bar.set_xticklabels([CAT_LABELS[c] for c in cat_vals], fontsize=9)
ax_bar.set_ylabel("Mean token count")
ax_bar.set_title("Average token count per category", fontsize=13, fontweight="bold")
ax_bar.legend(fontsize=9)
ax_bar.yaxis.get_major_locator().set_params(integer=True)

fig_bar.tight_layout()
fig_bar.savefig(OUTPUT_BARCHART, bbox_inches="tight")

# Print summary table to the terminal
header = f"{'Category':<16}" + "".join(f"{n:>10}" for n in TOKENIZERS)
print(header)
print("-" * len(header))
for j, cat in enumerate(cat_vals):
    row = f"{CAT_LABELS[cat]:<16}"
    row += "".join(f"{all_means[n][j]:>10.2f}" for n in TOKENIZERS)
    count = int((categories == cat).sum())
    print(f"{row}  (n={count})")

plt.show()