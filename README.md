# Identifying Domain-Specific Terminology with Large Language Models

Welcome to the repository for my thesis *'Identifying domain-specific terminology with large language models'*.

In this repository there are multiple folders. Down below I will give a short explanation of what each folder contains.

## Datasets

In the dataset folder is one dataset (`acronyms.json`) which contains the abbreviations and their unabbreviated versions. What is missing is the original dataset but that could not be included because of sensitivity reasons, also mentioned in the thesis.

## LLM Ranking

Contains two code files: `llm_ranking_gemma.py` and `llm_ranking_llama.py`. They were used to rank the terms based on their domain specificity. They ran on an external server, that is why the paths deviate from the other files. The setup between the LLMs differs, so that is why they have two separate code files. In the `outputs` folder are two `.csv` files containing the ranking of both models.

## Preprocessing

Contains the code file `preprocessing.py` which was used to preprocess all the data. The outcome of this could also not be shared because it was still in the form of sentences which were too sensitive.

## Results Section

Contains two code files: `llm_vs_human.py` and `tfidf_vs_human.py`. These files compare the expert annotations to the outcomes of both ranking methods. In the `outputs` folder are most of the figures and results used in the results section of the thesis.

## Survey

The survey folder contains `annotation_agreement_krippendorff.py`, used to check the agreement between the human evaluators. The `results` folder contains the results of the annotation agreement and the accompanying datasets of the survey. The `setup` folder contains an Excel file with the words and example sentences used in the survey. It also contains a README file which shows the questions used in the survey, in other words the survey setup.

## TF-IDF

Contains the code file `tfidf.py` with the code to run TF-IDF. The `outputs` folder contains the outcomes of TF-IDF and the `candidate_terms` used for the ranking of the LLMs, to make sure that TF-IDF and LLMs used the same terms for ranking.

## Tokenization

Contains two code files. `tokenization.py` tokenizes all the words from the Gemma ranking with five general purpose tokenizers and saves the results to the folder `outputs`. The second code file `visualize_tokens.py` puts all the visuals of this tokenization by category in the `outputs` folder.
