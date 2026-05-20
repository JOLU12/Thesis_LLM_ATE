# Imports
import polars as pl
import json

# Config
INPUT_CSV = "PATH/Thesis_deliverables_code/datasets/corpus_findings_text.csv"
ACRONYMS_JSON = "PATH/Thesis_deliverables_code/datasets/acronyms.json"
OUTPUT_CSV = "PATH/Thesis_deliverables_code/preprocessing/outputs/cleaned_sentences_wa.csv"

# Read dataset using polars
df = pl.read_csv(
    INPUT_CSV,
    infer_schema_length=0,
    try_parse_dates=False,
    ignore_errors=False,
    separator=";",
    quote_char=None,
    )

# Clean the text entries, by stripping numbers, punctuation and extra whitespaces
df = df.with_columns(
    pl.col("Text Entries")
        .str.to_lowercase()
        .str.replace_all(r"\b\d+\b", " ")        
        .str.replace_all(r"[^a-z\s]", " ")       
        .str.replace_all(r"\s+", " ")               
        .str.strip_chars()                         
        .alias("text_clean")
)

# Unfold the abbreviations that are in the dataset acronyms.json that was share by KLM
# and it was further expanded with abbreviations from the FAA abbreviations page
with open(ACRONYMS_JSON, "r") as f:
    acronyms = json.load(f)

expr = pl.col("text_clean")
for abbrev, full_word in acronyms.items():
    pattern = r"(?i)\b" + abbrev + r"\b"            # Make it case insensitive
    expr = expr.str.replace_all(pattern, full_word)

df = df.with_columns(expr.str.to_lowercase().alias("text_clean"))

df = df.select(pl.col("text_clean"))

df.write_csv(OUTPUT_CSV) # wa stands for "without abbreviations"


