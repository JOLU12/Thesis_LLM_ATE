import os
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import (
    roc_auc_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support,
)
import krippendorff
from collections import Counter

# Config
INPUT_ANNOTATIONS = "/Thesis_deliverables_code/survey/results/annotations_processed.csv"
INPUT_TFIDF = "/Thesis_deliverables_code/tfidf/outputs/tfidf_results.csv"
INPUT_LLAMA = "/Thesis_deliverables_code/llm_ranking/outputs/llm_ranking_llama.csv"
INPUT_GEMMA = "/Thesis_deliverables_code/llm_ranking/outputs/llm_ranking_gemma.csv"
OUTPUT_DIR = "/Thesis_deliverables_code/results_section/outputs"
OUTPUT_CSV = "/Thesis_deliverables_code/results_section/outputs/llm_vs_human.csv"
OUTPUT_TFIDF_DIST = "/Thesis_deliverables_code/results_section/outputs/tfidf_distribution.png"
OUTPUT_LLM_DIST = "/Thesis_deliverables_code/results_section/outputs/llm_distribution.png"
OUTPUT_HUMAN_DIST = "/Thesis_deliverables_code/results_section/outputs/human_distribution.png"

CATS_1_4 = [1, 2, 3, 4]
RATING_COLS = ["Rating 1", "Rating 2", "Rating 3"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def score_to_category(scores):
    """Map 0-100 domain scores to categories 1-5."""
    return np.select(
        [scores >= 80, scores >= 60, scores >= 40, scores >= 20],
        [1, 2, 3, 4],
        default=5,
    )


def bin_counts(scores, edges):
    """Count scores falling into each bin. Last bin is inclusive on the right."""
    counts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            counts.append(int(((scores >= lo) & (scores <= hi)).sum()))
        else:
            counts.append(int(((scores >= lo) & (scores < hi)).sum()))
    return counts


# Load the expert annotations 
df_ann = pl.read_csv(INPUT_ANNOTATIONS)
df_ann = df_ann.with_columns(pl.col("Word").str.to_lowercase().alias("word_lower"))
base = df_ann.drop_nulls(subset=["consensus"])

# Expert annotation krippendorff 
rating_matrix = df_ann.select(RATING_COLS).to_numpy().T.astype(float)
human_alpha = krippendorff.alpha(rating_matrix, level_of_measurement="ordinal")


# TF-IDF scores to compare with the LLM later
df_tfidf = pl.read_csv(INPUT_TFIDF)
df_tfidf = df_tfidf.with_columns(pl.col("term").str.to_lowercase().alias("term_lower"))

matched_tfidf = base.join(
    df_tfidf.select("term_lower", "tfidf_score"),
    left_on="word_lower", right_on="term_lower", how="inner",
)

tfidf_by_cat = [matched_tfidf.filter(pl.col("consensus") == c)["tfidf_score"].to_numpy()
                for c in CATS_1_4]
kruskal_h, _ = kruskal(*tfidf_by_cat)
n_matched = matched_tfidf.height
eta2_tfidf = (kruskal_h - len(CATS_1_4) + 1) / (n_matched - len(CATS_1_4))
rho_tfidf = abs(spearmanr(matched_tfidf["tfidf_score"].to_numpy(),
                           matched_tfidf["mean_rating"].to_numpy())[0])
auc1_tfidf = roc_auc_score((matched_tfidf["consensus"] == 1).cast(pl.Int8).to_numpy(),
                             matched_tfidf["tfidf_score"].to_numpy())
auc12_tfidf = roc_auc_score(matched_tfidf["consensus"].is_in([1, 2]).cast(pl.Int8).to_numpy(),
                              matched_tfidf["tfidf_score"].to_numpy())


# Each LLM model gets a colour for confusion matrix plots
llm_files = {
    "LLaMA Scout": (INPUT_LLAMA, "#E74C3C"),
    "Gemma4": (INPUT_GEMMA, "#27AE60"),
}

results = []
plot_data = []
score_distributions = {}

for name, (fpath, color) in llm_files.items():
    df_llm = pl.read_csv(fpath).drop_nulls(subset=["domain_score"])
    raw_scores = df_llm["domain_score"].to_numpy()
    score_distributions[name] = raw_scores

    df_llm = df_llm.with_columns(
        pl.col("term").str.to_lowercase().alias("term_lower"),
        pl.Series("llm_cat", score_to_category(raw_scores)),
    )

    merged = base.join(
        df_llm.select("term_lower", "domain_score", "llm_cat"),
        left_on="word_lower", right_on="term_lower", how="inner",
    )

    n = merged.height
    scores = merged["domain_score"].to_numpy()
    consensus = merged["consensus"].cast(pl.Int64).to_numpy()
    mean_rating = merged["mean_rating"].to_numpy()
    llm_cat = merged["llm_cat"].to_numpy()

    cm = confusion_matrix(consensus, llm_cat, labels=CATS_1_4)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True),
                        out=np.zeros_like(cm, dtype=float),
                        where=cm.sum(axis=1, keepdims=True) != 0)

    # Metrics
    H, _ = kruskal(*[scores[consensus == c] for c in CATS_1_4])
    eta2 = (H - len(CATS_1_4) + 1) / (n - len(CATS_1_4))
    rho = abs(spearmanr(scores, mean_rating)[0])
    auc1 = roc_auc_score((consensus == 1).astype(int), scores)
    auc12 = roc_auc_score(np.isin(consensus, [1, 2]).astype(int), scores)
    kappa = cohen_kappa_score(consensus, llm_cat, weights="linear", labels=CATS_1_4)
    exact = (consensus == llm_cat).mean()
    w1 = (np.abs(consensus - llm_cat) <= 1).mean()
    means = {c: float(scores[consensus == c].mean()) for c in CATS_1_4}
    correct_order = all(means[c] > means[c + 1] for c in CATS_1_4[:-1])

    # F1, Precision, Recall (macro + weighted)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        consensus, llm_cat, labels=CATS_1_4, average="macro", zero_division=0,
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        consensus, llm_cat, labels=CATS_1_4, average="weighted", zero_division=0,
    )

    # Krippendorff alpha with LLM as 4th rater
    r1, r2, r3 = [merged[c].to_numpy().astype(float) for c in RATING_COLS]
    llm_ratings = np.clip(llm_cat, 1, 4).astype(float)
    try:
        alpha_extended = krippendorff.alpha(
            np.array([r1, r2, r3, llm_ratings]), level_of_measurement="ordinal"
        )
    except Exception:
        alpha_extended = np.nan

    results.append(dict(
        Method=name, n=n,
        eta2=round(eta2, 4), rho=round(rho, 4),
        auc1=round(auc1, 4), auc12=round(auc12, 4),
        kappa=round(kappa, 4), exact=round(exact, 4), w1=round(w1, 4),
        alpha_with_llm=round(alpha_extended, 4),
        macro_f1=round(f1_mac, 4), macro_p=round(prec_mac, 4), macro_r=round(rec_mac, 4),
        weighted_f1=round(f1_wt, 4), weighted_p=round(prec_wt, 4), weighted_r=round(rec_wt, 4),
        correct_order=correct_order,
        means_cat1=round(means[1], 1), means_cat2=round(means[2], 1),
        means_cat3=round(means[3], 1), means_cat4=round(means[4], 1),
    ))

    plot_data.append(dict(
        name=name, color=color,
        cm=cm, cm_norm=cm_norm,
        rho=rho, kappa=kappa, exact=exact, w1=w1,
    ))

pl.DataFrame(results).write_csv(OUTPUT_CSV)


# Confusion matrix plot
X_LABELS = [
    "LLM Cat 1\n(80-100)\nIntra-subject",
    "LLM Cat 2\n(60-80)\nInter-subject",
    "LLM Cat 3\n(40-60)\nExtra-subject",
    "LLM Cat 4\n(20-40)\nNon-subject",
]
Y_LABELS = [
    "Human Cat 1\nIntra-subject",
    "Human Cat 2\nInter-subject",
    "Human Cat 3\nExtra-subject",
    "Human Cat 4\nNon-subject",
]
cmap = LinearSegmentedColormap.from_list("bw", ["#FFFFFF", "#1B4F72"])

for plot_item in plot_data:
    cm_counts = plot_item["cm"]
    cm_norm = plot_item["cm_norm"]
    color = plot_item["color"]
    name = plot_item["name"]
    model_slug = name.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#F8F9FA")

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")

    ax.set_xticks(range(4))
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(Y_LABELS, fontsize=9)
    ax.set_title(name, fontsize=11, fontweight="bold", color=color, pad=10)

    for i in range(4):
        for j in range(4):
            p = cm_norm[i, j]
            n_cell = cm_counts[i, j]
            cell_col = "white" if p > 0.55 else "#1B4F72"
            ax.text(
                j, i, f"{p:.2f}\n(n={n_cell})",
                ha="center", va="center", fontsize=9.5, color=cell_col,
                fontweight="bold" if i == j else "normal",
            )
            if i == j:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=color, linewidth=2.5,
                ))

    ax.text(
        0.5, -0.22,
        f"abs_rho={abs(plot_item['rho']):.3f}  kappa={plot_item['kappa']:.3f}"
        f"  Exact={plot_item['exact']:.1%}  Within-1={plot_item['w1']:.1%}",
        transform=ax.transAxes, ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9),
    )

    plt.suptitle(
        "(row = human label, col = LLM category | diagonal = exact match)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT_DIR + "/confusion_matrix_" + model_slug + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# Distribution plots
DIST_COLORS = {
    "tfidf": "#e05780",
    "gemma": "#2ec4b6",
    "llama": "#7c3aed",
    "human": "#f59e0b",
}

# TF-IDF score distribution
tfidf_scores = df_tfidf["tfidf_score"].to_numpy()
tfidf_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
tfidf_labels = [f"{tfidf_edges[i]}–{tfidf_edges[i+1]}" for i in range(len(tfidf_edges) - 1)]
tfidf_counts = bin_counts(tfidf_scores, tfidf_edges)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tfidf_labels, tfidf_counts, color=DIST_COLORS["tfidf"],
              edgecolor="white", linewidth=0.5)
for bar, cnt in zip(bars, tfidf_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            str(cnt), ha="center", va="bottom", fontsize=9, color="#555")
ax.set_xlabel("TF-IDF Score Range")
ax.set_ylabel("Number of Terms")
ax.set_title(f"TF-IDF Score Distribution (n = {len(tfidf_scores):,})")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(OUTPUT_TFIDF_DIST, dpi=200)
plt.close(fig)

# LLM domain score distributions (side-by-side)
llm_edges = [0, 20, 40, 60, 80, 101]
llm_labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]

model_names = list(score_distributions.keys())
model_counts = [bin_counts(score_distributions[m], llm_edges) for m in model_names]
model_colors = [color for _, (_, color) in llm_files.items()]

x = np.arange(len(llm_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
for i, (mname, mcounts, mcolor) in enumerate(zip(model_names, model_counts, model_colors)):
    offset = (i - (len(model_names) - 1) / 2) * width
    bar_set = ax.bar(x + offset, mcounts, width, label=mname,
                     color=mcolor, edgecolor="white", linewidth=0.5)
    for bar in bar_set:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10,
                str(int(h)), ha="center", va="bottom", fontsize=8, color="#555")

ax.set_xticks(x)
ax.set_xticklabels(llm_labels)
ax.set_xlabel("Domain Score Range")
ax.set_ylabel("Number of Terms")
n_strings = ", ".join(f"{m} n={len(score_distributions[m]):,}" for m in model_names)
ax.set_title(f"LLM Domain Score Distribution ({n_strings})")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_LLM_DIST, dpi=200)
plt.close(fig)

# Human expert category distribution
consensus_values = base["consensus"].cast(pl.Int64).to_list()
cat_counts = Counter(consensus_values)
cat_labels = ["Cat 1\nIntra-subject", "Cat 2\nInter-subject", "Cat 3\nExtra-subject", "Cat 4\nNon-subject"]
cat_values = [cat_counts.get(i, 0) for i in CATS_1_4]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cat_labels, cat_values, color=DIST_COLORS["human"],
              edgecolor="white", linewidth=0.5)
for bar, cnt in zip(bars, cat_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(cnt), ha="center", va="bottom", fontsize=10, color="#555")
ax.set_xlabel("Consensus Category")
ax.set_ylabel("Number of Terms")
ax.set_title(f"Expert Annotation Distribution (n = {sum(cat_values)})")
fig.tight_layout()
fig.savefig(OUTPUT_HUMAN_DIST, dpi=200)
plt.close(fig)