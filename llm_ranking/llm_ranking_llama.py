# Imports
import os
import json
import re
import glob
from datetime import datetime

import polars as pl
from llama_cpp import Llama
from carbontracker.tracker import CarbonTracker

# Configuration 
MODEL_DIR   = "/workspace/models/llama4-scout/Q4_K_S"
INPUT_CSV = "PATH/Thesis_deliverables_code/tfidf/outputs/candidate_terms.csv"
OUTPUT_CSV = "PATH/Thesis_deliverables_code/llm_ranking/outputs/llm_ranking_llama.csv"

BATCH_SIZE = 20
SEED = 42
CARBON_LOGS = "/workspace/outputs/carbon_logs"

# Find the first GGUF shard (llama-cpp auto-detects the rest)
gguf_files = sorted(glob.glob(MODEL_DIR + "/*.gguf"))
if not gguf_files:
    raise FileNotFoundError(f"No .gguf files found in {MODEL_DIR}")
MODEL_PATH = gguf_files[0]

# Load model
print("Loading model")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,   # offload all layers to GPU
    n_ctx=4096,      
    seed=SEED,
    verbose=False,
)
print("Model loaded")

# Same system prompt as Gemma-4 V3
SYSTEM_PROMPT = """You are an NLP expert assessing lexical domain-specificity for aviation maintenance, repair and overhaul (MRO) terminology.

Score each term from 0 to 100 based on its inherent lexical characteristics - morphological form, etymology, and how restricted its meningful usage is
across professional and genearl language contexts. You are assessing terms as they would appear in aviation maintenance documentation.

Scoring bands:
- 80-100: Aviation-specific (intra-subject terminology). The term's primary or most distinctive meaning belongs to aviation, aerospace, or MRO. Even if 
          recognizable to a general reader, its core technical meaning is aviation-owned.

- 60-80: Cross-domain technical (inter-subject terminology). The term has a recognized technical meaning shared across multiple engineering or scientific
         fields. It belongs to a technical register but not primarily aviation. 

- 40-60: Context dependent (extra-subject terminology). The term has no strong domain affiliation but takes on more precise meaning in professional or 
         technical contexts. A general reader understands the core word but not its precise technical application. 

- 20-40: General language (non-subject terminology). The term is used broadly across many domains and everyday language with no meaningful technical 
         specificity.

- 0-20: Non-scorable term. The term's meaning is genuinely unrecoverable without external context, it is a typo, an abbreviation whose referent cannot be 
        from the term alone, or an unresolvable merged fragment. 

Rules:
- Do NOT place morphologically irregular but semantically recoverable terms in the 0-20 band. Compound forms, closer compounds, and domain-specific shortenings
  should be scored by their semantic domain affiliation, not penalized for their morphological form. 
- Abbreviations score 0-20 only if their meaning is genuinely unrecoverable without external knowledge. Abbreviations with a recoverable referent score according
  to that referent's domain-specificity. 
- Use the full 20-point range within each band to express degree of fit. 
- Score each term independently.

Return ONLY a valid JSON array of objects with "term" and "score". No explanation, no markdown, not text before or after.
"""

def format_prompt(terms):
    n = len(terms)
    word_list = "\n".join(f'{i + 1}. term="{t}"'for i, t in enumerate(terms))
    return (
        f"SCore ALL {n} terms below from 0-100 according to the category bands."
        f"Use the exact term string. \n\n"
        f"{word_list}\n\n"
        f"Return ONLY a JSON array with \"term\" and \"score\" for every term above. No other text."
    )

def score_batch(terms):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": format_prompt(terms)},
        ],
        temperature=0.0,    # fully deterministic
        max_tokens=2048,
    )
    return response["choices"][0]["message"]["content"]

def extract_scores(raw):
    """Same parsing job as we did with Gemma-4 taking into account the same issues"""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    else:
        cleaned = "[" + cleaned + "]"

    raw_parsed = json.loads(cleaned)

    # Normalize the names
    scored_terms = []
    for entry in raw_parsed:
        term  = entry.get("term",  entry.get("word", ""))
        score = entry.get("score", entry.get("domain_score", entry.get("value", None)))
        if term and score is not None:
            scored_terms.append({"term": term, "domain_score": int(score)})
    return scored_terms

# Process all candidate terms
df = pl.read_csv(INPUT_CSV)
all_terms = df["term"].to_list()
results = []

n_batches = -(-len(all_terms) // BATCH_SIZE)

print(f"Total terms to classify: {len(all_terms)}")

start_time = datetime.now()
tracker = CarbonTracker(epochs=n_batches, components="gpu", log_dir=CARBON_LOGS)

for i in range(0, len(all_terms), BATCH_SIZE):
    batch_terms = all_terms[i: i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"[Batch {batch_num}/{n_batches}] rows {i}-{i + len(batch_terms) - 1}")

    tracker.epoch_start()

    try:
        raw = score_batch(batch_terms)
    except Exception as e:
        print(f"Error during generation: {e}")
        tracker.epoch_end()
        continue

    try:
        scored_terms = extract_scores(raw)
        results.extend(scored_terms)
        print(f"{len(scored_terms)}/{len(batch_terms)} terms scored")

        if len(scored_terms) != len(batch_terms):
            found_terms = {s["term"] for s in scored_terms}
            unscored    = set(batch_terms) - found_terms
            print(f"  Missing terms: {list(unscored)[:5]}")
    
    except json.JSONDecodeError as e:
        print(f"  ERROR: could not parse JSON for batch {batch_num}")
        print(f"  Raw output (first 300 chars):\n{raw[:300]}")
        print(f"  Raw output (last  300 chars):\n{raw[-300:]}")
    
    tracker.epoch_end()

# Join terms back to the orginal list
results_df    = pl.DataFrame(results)
merged        = df.join(results_df, on="term", how="left")
merged_sorted = merged.sort("domain_score", descending=True)
merged_sorted.write_csv(OUTPUT_CSV)

print(f"Done {len(results)}/{len(df)} terms scored")

tracker.stop()

end_time = datetime.now()
duration = end_time - start_time
hours, remainder = divmod(int(duration.total_seconds()), 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total time: {hours}h {minutes}m {seconds}s")
