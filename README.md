# Interpretable Quality Assessment for Machine
Translation: A Question-Answering Approach to
Semantic Consistency Evaluation

This directory contains three extensions to the [AskQE](https://github.com/dayeonki/askqe) framework, reimplemented using **Qwen2.5-3B-Instruct** as a smaller, open-weight alternative to LLaMA-3 70B.

All experiments are applied to two datasets: **BioMQM** (naturally occurring MT errors) and **ContraTICO** (synthetic perturbations).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clone the repository (if not already)
git clone https://github.com/Simone280802/AskQE_DNLP_2025-2026.git
cd AskQE_DNLP_2025-2026
```

> **GPU required.** All notebooks are designed for Kaggle (T4/P100) or Google Colab. Each notebook auto-detects the environment.

## Directory Structure

```
Qwen2.5-3B-Instruct/
├── requirements.txt
├── README.md                      ← You are here
├── biomqm/
│   ├── baseline/                  # Baseline QG → QA → Evaluation
│   ├── ner-extension/             # Extension 1: NER-enhanced pipeline
│   │   └── README.md
│   ├── prompt-ablation/           # Extension 2: Prompt strategy ablation
│   │   └── README.md
│   └── metrics-extension/         # Extension 3: NLI & LLM-Judge metrics
│       └── README.md
└── contratico/
    ├── baseline/                  # Baseline (ContraTICO)
    ├── ner-extension/             # Extension 1 (ContraTICO)
    ├── prompt-ablation/           # Extension 2 (ContraTICO)
    └── metrics-extension/         # Extension 3 (ContraTICO)
```

## Extensions Overview

| # | Extension | Goal | Key Idea |
|---|-----------|------|----------|
| 1 | **[NER Extension](biomqm/ner-extension/README.md)** | Improve question generation quality | Extract biomedical entities (`d4data/biomedical-ner-all`) and generate entity-aware questions |
| 2 | **[Prompt Ablation](biomqm/prompt-ablation/README.md)** | Find best prompting strategy | Compare P1-FewShot, P2-CoT, P3-Concise for QA |
| 3 | **[Metrics Extension](biomqm/metrics-extension/README.md)** | Richer evaluation metrics | Add NLI classifier + LLM-Judge for agreement analysis |

## Models Used

| Model | HuggingFace ID | Used in |
|-------|---------------|---------|
| Qwen 2.5 3B | `Qwen/Qwen2.5-3B-Instruct` | QG, QA, LLM Judge |
| Biomedical NER | `d4data/biomedical-ner-all` | NER Extraction |
| MiniLM SBERT | `sentence-transformers/all-MiniLM-L6-v2` | Semantic Similarity |
| BART MNLI | `facebook/bart-large-mnli` | NLI Classification |

## Execution Order

For a full reproduction, run the unified notebooks in this order:

1. **Baseline**
   - `biomqm/baseline/baseline.ipynb`
   - `contratico/baseline/baseline.ipynb`
2. **NER Extension**
   - `biomqm/ner-extension/ner_pipeline.ipynb`
   - `contratico/ner-extension/ner_pipeline.ipynb`
3. **Prompt Ablation**
   - `biomqm/prompt-ablation/prompt_ablation.ipynb`
   - `contratico/prompt-ablation/prompt_ablation.ipynb`
4. **Metrics Extension**
   - `biomqm/metrics-extension/metrics-extension.ipynb`
   - `contratico/metrics-extension/metrics-extension.ipynb`

Each notebook is self-contained and handles the entire pipeline for its respective section.
