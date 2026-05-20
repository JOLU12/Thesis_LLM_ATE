import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

TFIDF_COLOR = "#e05780"

# Config
INPUT_TFIDF = "/Thesis_deliverables_code/tfidf/outputs/tfidf_results.csv"
INPUT_ANNOTATIONS = "/Thesis_deliverables_code/survey/results/annotations_processed.csv"
OUTPUT_PNG = "/Thesis_deliverables_code/results_section/outputs/tfidf_confusion_matrix.png"
OUTPUT_CSV = "/Thesis_deliverables_code/results_section/outputs/tfidf_vs_human_merged.csv"

df_tfidf = pl.read_csv(INPUT_TFIDF)
df_ann = pl.read_csv(INPUT_ANNOTATIONS)

df_ann = df_ann.with_columns(
    pl.col("Word").str.to_lowercase().alias("word_lower")
)
df_tfidf = df_tfidf.with_columns(
    pl.col("term").str.to_lowercase().alias("term_lower")
)

# We only keep terms that have a valid consensus label from the experts
base = df_ann.drop_nulls(subset=["consensus"])
merged = base.join(
    df_tfidf.select("term_lower", "tfidf_score"),
    left_on="word_lower",
    right_on="term_lower",
    how="inner",
)
cats = sorted(merged["consensus"].drop_nulls().unique().to_list())

# split the TF-IDF score by human category to see whether the distributions differ
groups = [
    merged.filter(pl.col("consensus") == c)["tfidf_score"].to_numpy()
    for c in cats
]

H, p = kruskal(*groups)
eta2 = (H - len(cats) + 1) / (merged.height - len(cats))

rho, p_rho = spearmanr(
    merged["tfidf_score"].to_numpy(),
    merged["mean_rating"].to_numpy(),
)

auc1 = roc_auc_score(
    (merged["consensus"] == 1).cast(pl.Int8).to_numpy(),
    merged["tfidf_score"].to_numpy(),
)
auc12 = roc_auc_score(
    merged["consensus"].is_in([1, 2]).cast(pl.Int8).to_numpy(),
    merged["tfidf_score"].to_numpy(),
)

print(f"Kruskal-Wallis H={H:.3f}, p={p:.4f}")
print(f"eta2={eta2:.4f}")
print(f"Spearman rho={rho:.4f}, p={p_rho:.4f}")
print(f"AUC Cat1={auc1:.4f}")
print(f"AUC Cat1+2={auc12:.4f}")

desc = (
    merged.group_by("consensus")
    .agg(
        pl.col("tfidf_score").count().alias("count"),
        pl.col("tfidf_score").mean().alias("mean"),
        pl.col("tfidf_score").std().alias("std"),
        pl.col("tfidf_score").min().alias("min"),
        pl.col("tfidf_score").quantile(0.25).alias("25%"),
        pl.col("tfidf_score").median().alias("50%"),
        pl.col("tfidf_score").quantile(0.75).alias("75%"),
        pl.col("tfidf_score").max().alias("max"),
    )
    .sort("consensus")
)
print(desc)

# Map continueous TF-IDF scores into the same 4 category scheme as the annotations
# to be able to build a confusion matrix. 
scores = merged["tfidf_score"].to_numpy()

def score_to_category(score):
    if score >= 75:
        return 1
    elif score >= 50:
        return 2
    elif score >= 25:
        return 3
    else:
        return 4


tfidf_cat = np.array([score_to_category(s) for s in scores])
human_cat = merged["consensus"].to_numpy()

labels = [1, 2, 3, 4]

cm = confusion_matrix(human_cat, tfidf_cat, labels=labels)
cm_norm = np.divide(
    cm, cm.sum(axis=1, keepdims=True),
    out=np.zeros_like(cm, dtype=float),
    where=cm.sum(axis=1, keepdims=True) != 0,
)
print(cm)

print(classification_report(
    human_cat, tfidf_cat,
    labels=labels,
    target_names=["Intra-subject", "Inter-subject", "Extra-subject", "Non-subject"],
    digits=3, zero_division=0,
))

kappa = cohen_kappa_score(human_cat, tfidf_cat, weights=None)
kappa_linear = cohen_kappa_score(human_cat, tfidf_cat, weights="linear")
exact = np.mean(human_cat == tfidf_cat)
w1 = np.mean(np.abs(human_cat - tfidf_cat) <= 1)
print(f"Cohen's kappa={kappa:.3f}, linear={kappa_linear:.3f}, exact={exact:.1%}, within-1={w1:.1%}")

# Confusion matrix visualization 
X_LABELS = [
    "TF-IDF Cat 1\n(75-100)\nIntra-subject",
    "TF-IDF Cat 2\n(50-75)\nInter-subject",
    "TF-IDF Cat 3\n(25-50)\nExtra-subject",
    "TF-IDF Cat 4\n(0-25)\nNon-subject",
]
Y_LABELS = [
    "Human Cat 1\nIntra-subject",
    "Human Cat 2\nInter-subject",
    "Human Cat 3\nExtra-subject",
    "Human Cat 4\nNon-subject",
]
cmap = LinearSegmentedColormap.from_list("bw", ["#FFFFFF", "#1B4F72"])

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("#F8F9FA")

im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")

ax.set_xticks(range(4))
ax.set_xticklabels(X_LABELS, fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels(Y_LABELS, fontsize=9)
ax.set_title("TF-IDF", fontsize=11, fontweight="bold", color=TFIDF_COLOR, pad=10)

for i in range(4):
    for j in range(4):
        prop = cm_norm[i, j]
        n_cell = cm[i, j]
        cell_col = "white" if prop > 0.55 else "#1B4F72"
        ax.text(
            j, i, f"{prop:.2f}\n(n={n_cell})",
            ha="center", va="center", fontsize=9.5, color=cell_col,
            fontweight="bold" if i == j else "normal",
        )
        if i == j:
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=TFIDF_COLOR, linewidth=2.5,
            ))

ax.text(
    0.5, -0.22,
    f"abs_rho={abs(rho):.3f}  kappa={kappa_linear:.3f}"
    f"  Exact={exact:.1%}  Within-1={w1:.1%}",
    transform=ax.transAxes, ha="center", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=TFIDF_COLOR, alpha=0.9),
)

plt.suptitle(
    "(row = human label, col = TF-IDF category, diagonal = exact match)",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
plt.close(fig)

merged_out = merged.with_columns(pl.Series("tfidf_category", tfidf_cat))
merged_out.write_csv(OUTPUT_CSV)