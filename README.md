# AI-and-Sustainability

---

This repository provides a full pipeline for studying how companies talk about  Artificial Intelligence (AI) within the context of the United Nations Sustainable Development Goals (SDGs).
It takes raw corporate sustainability reports, converts them into clean sentences, filters for AI‑related content, classifies the sentences into SDGs and sentiment using OpenAI models, and produces aggregated statistics and visualizations.

## Getting Started

---

Python environment:

```python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh   # or: export PYTHONPATH=.:$PYTHONPATH
```

Set your OpenAI API key: ```export OPENAI_API_KEY="your_key"```


## Pipeline

---

### Extract text from reports
Convert PDF sustainability reports into plain text while preserving layout as much as possible.
Scripts can optionally invoke OCR for documents that are scans or have problematic encoding, ensuring no relevant content is lost.
Run the following:

```python
python src/data_processing/pdf2text.py
python src/data_processing/tesseract.py  # optional OCR
```

### Split and clean sentences

The raw text is split into sentences and scrubbed of stray characters, duplicate whitespace, and other artifacts.
If the reports are in languages other than English, translation scripts can convert them into English to maintain consistency. Run:

```python
python src/data_processing/splitter.py
python src/data_processing/translate.py  # optional translation
```

### Filter candidate sentences
Candidate sentences are ranked by semantic similarity against reference AI and SDG vectors, giving a probabilistic measure of relevance.
For edge cases or quick baselines, the pipeline can fall back to fuzzy keyword search to avoid missing obvious hits.
Run:

```python
python src/filtering/embedding_filter.py
python src/filtering/fuzzy_search.py
```

### Batch classification with OpenAI
Large sets of sentences are bundled into batches to efficiently submit them to OpenAI for sentiment and SDG classification.
The workflow includes scripts for creating jobs, polling their completion, and extracting the structured results.
Run:

```python
python src/classification/batch_requests.py
python src/classification/poll_requests.py
python src/classification/extract_results.py
```

### Aggregate & analyze
Once classifications are complete, the pipeline aggregates sentiment and SDG scores per company, year, and goal.
Additional plotting scripts generate quick visuals to spot trends or anomalies across the dataset.
Run:

```python
python src/analysis/generate_scores.py
python src/plots/sdg_sentiments.py
```

### Results
The pipeline generates the following artifacts:

- **Sentence-level scores**
  `data/scores_csv/.../similarity_scores.csv` – similarity scores at the sentence level.

- **Raw model outputs**
  `src/classification/results/merged_classifications.json` – unprocessed classification results.

- **Aggregated results**
  `src/classification/results/company_year_sentiment_counts.csv` – aggregated SDG/AI counts grouped by *company*, *year*, and *sentiment*.

- **Visualizations**:

  Plots such as *positive vs. negative counts per SDG*.