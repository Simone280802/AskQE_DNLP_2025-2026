# Interpretable Quality Assessment for Machine Translation: A Question-Answering Approach to Semantic Consistency Evaluation

## Overview

This repository presents **three novel extensions** to the [AskQE framework](https://github.com/dayeonki/askqe) (Kim et al., 2024), reimplemented using **Qwen2.5-3B-Instruct** as a smaller, open-weight alternative to proprietary LLaMA-3 70B models. 

**Core Idea**: Machine translation quality is assessed by generating questions from source text, answering them on translated text, and measuring semantic consistency through answer agreement.

### Scientific Motivation

Traditional MT evaluation metrics (BLEU, COMET) provide scalar scores but lack interpretability. AskQE introduces **explainability** through natural language questions that pinpoint specific semantic errors. Our work:

1. **Democratizes** the approach with smaller, open models (3B vs 70B parameters)
2. **Enhances** question generation via biomedical entity recognition
3. **Evaluates** multiple prompting strategies systematically
4. **Extends** evaluation metrics beyond exact match to include NLI and LLM-based agreement

## Key Contributions

### Extension 1: NER-Enhanced Question Generation
- **Problem**: Generic questions may miss domain-specific semantic shifts
- **Solution**: Extract biomedical entities using `d4data/biomedical-ner-all` and generate entity-focused questions


### Extension 2: Prompt Strategy Ablation
- **Problem**: Unclear which prompting approach yields best QA performance
- **Solution**: Systematic comparison of three strategies:
  - **P1-FewShot**: 3-shot examples
  - **P2-CoT**: Chain-of-thought reasoning
  - **P3-Concise**: Minimal instructions

### Extension 3: Multi-Metric Evaluation
- **Problem**: Binary exact-match misses semantic equivalence
- **Solution**: Add NLI classifier (BART-MNLI) and LLM-Judge for nuanced agreement scoring


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
