# Extension 1: NER-Enhanced AskQE Pipeline

This extension enriches question generation with **Named Entity Recognition**. Instead of generic questions, the pipeline extracts biomedical entities using `d4data/biomedical-ner-all` and generates entity-specific questions that better capture domain-critical information.

## Pipeline Overview

```
Source Text
    │
    ▼
┌─────────────────────┐
│  1. NER Extraction   │  d4data/biomedical-ner-all
│     → entities.jsonl │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Entity-Aware QG  │  Qwen2.5-3B-Instruct
│     → qg_output.jsonl│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. QA Source        │  Qwen2.5-3B-Instruct
│     → source.jsonl   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. QA BT (×5 langs)│  Qwen2.5-3B-Instruct
│     → bt-{lang}.jsonl│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. Mapping          │  Combine source + BT answers
│     → mapped.jsonl   │
└──────────┬──────────┘
           ▼
┌────────────┬────────────┐
│ 6. String  │ 7. SBERT   │
│ Comparison │ Similarity │
└────────────┴────────────┘
```

## Directory Structure

```
ner-extension/
├── README.md                    ← You are here
├── ner_pipeline.ipynb           # Main notebook (run this)
├── code/
│   ├── ner_extraction.py        # Step 1: NER with BioBERT
│   ├── qg_entity_aware.py       # Step 2: Entity-aware QG
│   ├── qa_entity.py             # Steps 3-4: QA (source + BT)
│   ├── mapping.py               # Step 5: Combine answers
│   ├── string_comparison.py     # Step 6: F1/EM/chrF/BLEU
│   ├── sbert.py                 # Step 7: Cosine similarity
│   └── utils.py                 # Shared utilities
├── QG/                          # Question generation output
├── QA/                          # QA answers (source + BT)
└── evaluation/                  # Evaluation results
```

## Pipeline Steps

### 1. NER Extraction
We use `d4data/biomedical-ner-all` to extract biomedical entities (e.g., diseases, chemicals, anatomical structures) from the source text.
- **Script**: `code/ner_extraction.py`
- **Output**: `QG/ner_output.jsonl`

### 2. Entity-Aware Question Generation
Questions are generated specifically about the extracted entities, ensuring the model focuses on critical domain information.
- **Script**: `code/qg_entity_aware.py`
- **Output**: `QG/qg_entity_aware.jsonl`

### 3. Question Answering
We generate answers using the Qwen model for both the source text and the backtranslated text (all 5 languages).
- **Script**: `code/qa_entity.py`
- **Output**: `QA/unique/source.jsonl`, `QA/unique/bt-{lang}.jsonl`

### 4. Mapping & Evaluation
We map the answers and compute semantic similarity (SBERT) and string metrics (F1, EM, BLEU, chrF) to quantify the divergence between source and backtranslation answers.
- **Scripts**: `code/mapping.py`, `code/sbert.py`, `code/string_comparison.py`

## Usage

The entire pipeline is encapsulated in a single notebook:

**[ner_pipeline.ipynb](ner_pipeline.ipynb)**

Open this notebook in Jupyter, Google Colab, or Kaggle. It will:
1. Setup the environment and download models
2. Run NER extraction
3. Generate entity-aware questions
4. Generate answers for source and backtranslations
5. Compute all evaluation metrics

### ContraTICO Variant

The same pipeline is available for the **ContraTICO** dataset (synthetic errors) in:
- `../../contratico/ner-extension/ner_pipeline.ipynb`

It follows the same logic but uses ContraTICO's specific languages (es, fr, hi, tl, zh) and perturbation types.

