# RAG ESG

A collection of utilities for processing company annual reports, retrieving ESG-related passages with retrieval-augmented generation (RAG) techniques, running sentiment analysis, and summarising results.

## Data Processing

### `pdf2text.py`
Extracts text directly from PDFs using [PyMuPDF](https://pymupdf.readthedocs.io). The script walks through `data/reports/COMPANY/YEAR.pdf`, converts each file to text, and saves the output under `data/texts/COMPANY/YEAR/results.txt`.

### `tesseract.py`
Performs OCR on PDFs that cannot be parsed normally. Each page is rendered to an image with `pdf2image`, then processed by `pytesseract`; results are written alongside the original PDF in `data/reports_corrupted`.

## Retrieval-Augmented Generation (RAG)

### `splitter.py`
Cleans raw text, removes artifacts such as page numbers, and splits documents into individual sentences stored as `splits.json` files under `data/texts`.

### `embedding_retriever.py`
Uses the `all-MiniLM-L6-v2` SentenceTransformer to embed sentences and a set of SDG and AI keywords, generating a `sentence_scores.csv` file with cosine-similarity scores for each chunk.

### `fuzzy_search.py`
Augments `sentence_scores.csv` with RapidFuzz scores for several AI terms, marking sentences that exceed a configurable fuzzy-matching threshold.

### `embedding_filter.py`
Filters chunks whose embedding or fuzzy scores exceed a threshold and stores the surviving sentences in `embedding_filter.json` for each company/year pair.

### `rag_filter.py`
Loads filtered sentences, indexes them in Weaviate using OpenAI embeddings, and retrieves passages similar to predefined AI and SDG keywords. Results are saved as `rag_filter_ai.json` and `rag_filter_sdg.json`.

## Sentiment

### `classify_sentiment.py`
Demonstrates sentiment classification with a DistilBERT model fine-tuned on SST-2, returning the predicted label for a sample string.

## Analysis

### `embedding_scores.py`
Aggregates AI and SDG scores from all `sentence_scores.csv` files using DuckDB, producing a per-company, per-year CSV of embedding-based counts.

### `missing_files.py`
Checks for missing yearly PDFs and missing downstream RAG outputs, writing reports to `results/missing_data.csv` and `results/filtered_data.csv`.

### `rag_scores.py`
Summarises retrieved document counts for AI and SDG keywords across multiple similarity thresholds, saving separate result files for each threshold.

### `rag_individual_scores.py`
Generates per-SDG and AI scores per company/year for different thresholds, outputting to the `results` directory.

---

This README summarises the repository’s structure and key scripts for processing and analysing ESG-related documents.
