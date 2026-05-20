# imports
import polars as pl
import numpy as np
import krippendorff
from statistics import mode

# Config
INPUT_XLSX = "PATH/Thesis_deliverables_code/survey/results/domain_expert_annotation_structured.xlsx"
OUTPUT_CSV = "PATH/Thesis_deliverables_code/survey/results/annotations_processed.csv"

RATING_COLS = ["Rating 1", "Rating 2", "Rating 3"]

df = pl.read_excel(INPUT_XLSX)

for col in RATING_COLS:
    counts = df.group_by(col).len().sort(col)
    print(f"  {col}: {dict(zip(counts[col].to_list(), counts['len'].to_list()))}")

# Replace all 5 scores with null, which means the annotator could not assign a term
# so it is not used as a real rating in the agreement calculation
df = df.with_columns(
    pl.col(col).replace(5, None).alias(col)    
    for col in RATING_COLS
)

null_count = sum(df[col].null_count() for col in RATING_COLS)
fully_null_rows = df.filter(
    pl.all_horizontal(pl.col(col).is_null() for col in RATING_COLS)
).height

# Krippendorff's alpha on the ordinal scale, and this accounts for the fact that 
# disagreeing by 2 categories is wrose than disagreeing by one
rating_matrix = df.select(RATING_COLS).to_numpy().T.astype(float)

ordinal_alpha = krippendorff.alpha(rating_matrix, level_of_measurement="ordinal")
print(f"\nKrippendorff's alpha:  {ordinal_alpha:.4f}")

def majority_vote(row):
    """We take the majority vote (mode) as the consensus between the annotators"""
    vals = [row[c] for c in RATING_COLS if row[c] is not None]
    if not vals:
        return None
    return mode(vals)


df = df.with_columns(
    pl.struct(RATING_COLS)
    .map_elements(majority_vote, return_dtype=pl.Int64)
    .alias("consensus")
)

# Also store the arithmetic mean for correlation analyses later
df = df.with_columns(
    pl.mean_horizontal(pl.col(col) for col in RATING_COLS).alias("mean_rating")
)

print("\nConsensus label distribution:")
dist = df.group_by("consensus").len().sort("consensus")
print(dist)
print(f"Excluded (all raters null): {df['consensus'].null_count()}")


# Flag terms where raters were more than 2 categories apart 
df = df.with_columns(
    (
        pl.max_horizontal(pl.col(col) for col in RATING_COLS)
        - pl.min_horizontal(pl.col(col) for col in RATING_COLS)
    ).alias("max_disagreement")
)

print("\nTerms with max spread >= 3 between raters:")
disagreements = df.filter(pl.col("max_disagreement") >= 3).select(
    ["Word", *RATING_COLS, "consensus"]
)
print(disagreements)

df.write_csv(OUTPUT_CSV)