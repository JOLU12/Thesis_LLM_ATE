# Imports
import math
import re
import time
from collections import Counter

import polars as pl
import spacy
from datasets import load_dataset

# Config
INPUT_PATH = "PATH/Thesis_deliverables_code/preprocessing/outputs/cleaned_sentences_wa.csv"
OUTPUT_PATH = "PATH/Thesis_deliverables_code/tfidf/outputs/tfidf_results.csv"
CANDIDATE_PATH = "PATH/Thesis_deliverables_code/tfidf/outputs/candidate_terms.csv"

# Minimum number of Wikipedia articles a term must appear in to count as in general language
MIN_DF = 5    

# We only want nouns, proper nouns, and adjectives for domain term extraction
# Verbs and function words did not need to be passed because they are not domain specific
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS = {"NOUN", "PROPN", "ADJ"}

def extract_tokens(text):
    """Extract lemmatized nouns/adjectives from the aviation corpus.
    Processes in 100k-character chunks becuase spaCy chokes on very long strings."""
    tokens = []
    for start in range(0, len(text), 100_000):
        doc = nlp(text[start:start + 100_000])
        for token in doc:
            if token.is_stop or not token.is_alpha or len(token) <= 2:
                continue
            if token.pos_ not in KEEP_POS:
                continue
            tokens.append(token.lemma_.lower())
    return tokens

# Wikipedia corpus is very big so we needed a fast regex tokenizer instead of spaCy
# Only need unique token sets for document frequency not POS tags
WORD_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z]+\b")

def wiki_word_set(text):
    return set(WORD_RE.findall(text.lower()))


def load_corpus(filepath):
    """Concatenate all cleaned maintenance findings into one big string and extra lemmatized domain terms"""
    df = pl.read_csv(filepath)
    if "text_clean" not in df.columns:
        raise ValueError(
            f"CSV must have column 'text_clean', found: {df.columns}"
        )
    text = " ".join(df["text_clean"].drop_nulls().to_list())
    return extract_tokens(text)


def calc_doc_freq(domain_vocab, batch_size=10_000):
    """Steam the full English Wikipedia dump to count how many articles contain each of our domain terms"""
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
 
    doc_freq = Counter()
    total_docs = 0
    batch_terms = []
 
    for article in dataset:
        terms = wiki_word_set(article["text"]) & domain_vocab
        batch_terms.append(terms)
        total_docs += 1
 
        if len(batch_terms) >= batch_size:
            for term_set in batch_terms:
                doc_freq.update(term_set)
            batch_terms = []
 
            if total_docs % 100_000 == 0:
                print(f"Processed {total_docs}")
 
    # flush remaining
    for term_set in batch_terms:
        doc_freq.update(term_set)
 
    return doc_freq, total_docs


def build_tfidf_table(term_counts, doc_freq, total_docs, min_df=MIN_DF):
    # if a term never appears in wikipedia at all, it gets max IDF
    idf_max = math.log((1 + total_docs) / 1.0) + 1
 
    rows = []
    for term, count in term_counts.items():
        df_val = doc_freq.get(term, 0)
        tf = 1.0 + math.log(count) if count > 0 else 0.0
 
        if df_val == 0:
            idf = idf_max
            in_wikipedia = False
        else:
            idf = math.log((1 + total_docs) / (1 + df_val)) + 1
            in_wikipedia = True
 
        rows.append({
            "term": term,
            "raw_count_domain": count,
            "tf": round(tf, 6),
            "df_wikipedia": df_val,
            "idf": round(idf, 6),
            "tfidf_score": round(tf * idf, 6),
            "in_wikipedia": in_wikipedia,
            "below_min_df": 0 < df_val < min_df,
        })
 
    return pl.DataFrame(rows).sort("tfidf_score", descending=True)


def main():
    t_start = time.perf_counter()
 
    print("Loading and tokenizing domain corpus")
    tokens = load_corpus(INPUT_PATH)
    term_counts = Counter(tokens)
    domain_vocab = set(term_counts.keys())
    print(f"{len(tokens):,} tokens, {len(domain_vocab):,} unique terms")
 
    # save the raw term list — this is what gets fed into the LLM ranking scripts
    candidate_df = pl.DataFrame({"term": sorted(term_counts.keys())})
    candidate_df.write_csv(CANDIDATE_PATH)
    print("Saved candidate term list")
 
    print("Streaming Wikipedia corpus for document frequencies")
    doc_freq, total_docs = calc_doc_freq(
        domain_vocab=domain_vocab,
    )
    print(f"{total_docs:,} articles processed, {len(doc_freq):,} domain terms found in Wikipedia")
 
    print("Computing TF-IDF scores")
    results = build_tfidf_table(
        term_counts=term_counts,
        doc_freq=doc_freq,
        total_docs=total_docs,
        min_df=MIN_DF,
    )
 
    results.write_csv(OUTPUT_PATH)
 
    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.2f} seconds. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()