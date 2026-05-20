# Imports 
import numpy as np
import polars as pl
from transformers import AutoTokenizer
import tiktoken

# Config
HF_TOKEN = "TOKEN"
INPUT_CSV = "PATH/Thesis_deliverables_code/llm_ranking/outputs/llm_ranking_gemma.csv"
OUTPUT_CSV = "PATH/Thesis_deliverables_code/tokenization/outputs/llm_ranking_gemma_tokenized.csv"


# We tokenize the Gemma-ranked terms with five different tokenizers to see
# Whether domain-specific words get split into more subword pieces than general vocabulary

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

llama4_tok = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    token=HF_TOKEN,
)

gemma4_tok = AutoTokenizer.from_pretrained(
    "google/gemma-4-31B",
    token=HF_TOKEN,
)

openai_tok = tiktoken.get_encoding("o200k_base")

mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")


def run_tokenizer(tokenizer, words):
    """Run a huggingface tokenizer over a list of words that returns subword counts 
    and the actual pieces"""
    tokenized = [tokenizer.tokenize(str(w)) for w in words]
    counts = [len(t) for t in tokenized]
    strings = [" | ".join(t) for t in tokenized]
    return counts, strings


def run_tiktoken(encoding, words):
    """Same thing as above but than for OpenAI tiktoken, but the API is slightly
    different because they work with integer IDs instead of string tokens"""
    counts = []
    strings = []
    for w in words:
        token_ids = encoding.encode(str(w))
        counts.append(len(token_ids))
        token_strs = [encoding.decode([tid]) for tid in token_ids]
        strings.append(" | ".join(token_strs))
    return counts, strings


def score_to_category(scores):
    """Map the continuous domain scores of the LLM ranking steps into the 5 categories
    used through the thesis"""
    return np.select(
        [scores >= 80, scores >= 60, scores >= 40, scores >= 20],
        [1, 2, 3, 4],
        default=5,
    )


# Load the data and run all terms through all five tokenziers
df = pl.read_csv(INPUT_CSV)
terms = df["term"].to_list()
categories = score_to_category(df["domain_score"].to_numpy())

bert_counts, bert_strs = run_tokenizer(bert_tok, terms)
llama4_counts, llama4_strs = run_tokenizer(llama4_tok, terms)
gemma4_counts, gemma4_strs = run_tokenizer(gemma4_tok, terms)
openai_counts, openai_strs = run_tiktoken(openai_tok, terms)
mistral_counts, mistral_strs = run_tokenizer(mistral_tok, terms)

# Keep the column "term" and add domain category with new token columns
df = df.select(["term", "raw_count_domain"]).with_columns([
    pl.Series("category", categories),
    pl.Series("bert_token_count", bert_counts),
    pl.Series("bert_tokens", bert_strs),
    pl.Series("llama4_token_count", llama4_counts),
    pl.Series("llama4_tokens", llama4_strs),
    pl.Series("gemma4_token_count", gemma4_counts),
    pl.Series("gemma4_tokens", gemma4_strs),
    pl.Series("openai_token_count", openai_counts),
    pl.Series("openai_tokens", openai_strs),
    pl.Series("mistral_token_count", mistral_counts),
    pl.Series("mistral_tokens", mistral_strs),
])

df.write_csv(OUTPUT_CSV)
