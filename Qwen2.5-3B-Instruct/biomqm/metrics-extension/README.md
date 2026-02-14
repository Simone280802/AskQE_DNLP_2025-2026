# Extension 3: Metrics Extension (NLI & LLM Judge)

This extension adds two new evaluation approaches beyond SBERT and string comparison:

| Metric | Model | Purpose |
|--------|-------|---------|
| **NLI Classifier** | `facebook/bart-large-mnli` | Classify answer pairs as entailment/neutral/contradiction |
| **LLM Judge** | `Qwen/Qwen2.5-3B-Instruct` | Use the LLM itself as an NLI judge for comparison |

The NLI and LLM Judge outputs are then compared via:
- **Agreement Rate** — How often NLI and LLM Judge agree, by severity/perturbation
- **Confidence Analysis** — Average confidence when assigning labels and when agreeing
- **Confusion Matrix** — NLI vs LLM Judge label distribution

## Pipeline Overview

```
Mapped answers (baseline mapping.jsonl)
    │
    ├──────────────────────────────┐
    ▼                              ▼
┌──────────────┐           ┌────────────┐
│ NLI Classifier│           │ LLM Judge  │
│ (BART-MNLI)  │           │ (Qwen 3B)  │
└──────┬───────┘           └──────┬─────┘
       │                          │
       └────────────┬─────────────┘
                    ▼
        ┌───────────────────────┐
        │ Post-hoc Analysis     │
        │ • Agreement Rate      │
        │ • Confidence          │
        │ • Confusion Matrix    │
        └───────────────────────┘
```

## Directory Structure

```
metrics-extension/
├── README.md                          ← You are here
├── metrics-extension.ipynb            # Main notebook (run this)
├── evaluation/
│   ├── nli/nli_classifier.py          # NLI classification
│   └── llm-judge/llm_judge.py         # LLM Judge
└── results/
    ├── nli/{lang}-nli.jsonl           # NLI per-language output
    ├── llm-judge/{lang}-llm-judge.jsonl # LLM Judge output
    ├── AGREEMENT_RATE-NLI_LLM/        # Agreement analysis
    ├── CONFIDENCE - NLI_LLM/          # Confidence analysis
    └── CONFUSION_MATRIX - NLI_LLM/    # Confusion matrices
```

## Pipeline Steps

### 1. Load Baseline Data
The pipeline starts by loading the mapped QA data from the baseline (`biomqm/baseline/mapping.jsonl`).

### 2. NLI Classification
We use `facebook/bart-large-mnli` to classify each (source answer, backtranslated answer) pair as Entailment, Neutral, or Contradiction.
- **Script**: `evaluation/nli/nli_classifier.py`
- **Output**: `results/nli/{lang}-nli.jsonl`

### 3. LLM Judge
We prompt Qwen2.5-3B-Instruct to act as a judge and classify the same pairs, providing reasoning.
- **Script**: `evaluation/llm-judge/llm_judge.py`
- **Output**: `results/llm-judge/{lang}-llm-judge.jsonl`

## Usage

The entire pipeline is encapsulated in a single notebook:

**[metrics-extension.ipynb](metrics-extension.ipynb)**

Open this notebook in Jupyter, Google Colab, or Kaggle. It will:
1. Setup the environment and download models
2. Verify input data availability
3. Run NLI classification
4. Run LLM Judge evaluation

### ContraTICO Variant

The same pipeline is available for the **ContraTICO** dataset (synthetic errors) in:
- `../../contratico/metrics-extension/metrics-extension.ipynb`

It follows the same logic but analyzes results by **perturbation type** rather than severity.


## Post-hoc Analysis (BioMQM)

After running the notebook, use the analysis scripts in `results/` to generate agreement, confidence, and confusion matrix reports:

```python
# Agreement Rate (by severity)
python "results/AGREEMENT_RATE-NLI_LLM/agreement_rate_by_severity.py"

# Confidence Analysis
python "results/CONFIDENCE - NLI_LLM/confidence.py"

# Confusion Matrix
python "results/CONFUSION_MATRIX - NLI_LLM/confusion_matrix.py"
```

## ContraTICO Variant

The ContraTICO version is in `../../contratico/metrics-extension/`. Key differences:
- **Analysis grouping**: by **perturbation type** instead of severity
- **Pipelines**: separate analysis for `atomic`, `semantic`, `vanilla` configs
- **Languages**: es, fr, hi, tl, zh
- **Scripts**: `nli_classifier_contratico.py` and `llm_judge_contratico.py`
- **Post-hoc scripts** in `results/`: `agreement_rate.py`, `confidence.py`, `confusion_matrix.py` (one per pipeline)
