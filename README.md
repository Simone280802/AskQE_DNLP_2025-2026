# Interpretable Quality Assessment for Machine Translation: A Question-Answering Approach to Semantic Consistency Evaluation

## Overview

This repository presents **three novel extensions** to the [AskQE framework](https://github.com/dayeonki/askqe) (Kim et al., 2024), reimplemented using **Qwen2.5-3B-Instruct** as a smaller, open-weight alternative to proprietary LLaMA-3 70B models. 

**Core Idea**: Machine translation quality is assessed by generating questions from source text, answering them on translated text, and measuring semantic consistency through answer agreement.

### Scientific Motivation

Traditional MT metrics (BLEU, F1) produce scalar scores without explaining translation errors. AskQE introduces explainability by generating natural language questions that pinpoint specific semantic inconsistencies between source and target texts. Our work:

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

### Extension 3: Metrics Extension
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
> **[Our project folder](Qwen2.5-3B-Instruct/)**

```
Qwen2.5-3B-Instruct/
├── requirements.txt
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



# Installation

**Prerequisites**
- Python 3.8+
- GPU with CUDA support (T4, P100, V100, or A100 recommended)
- PyTorch with CUDA
- Hugging Face Account: Access to Qwen2.5-3B-Instruct model

**Setup**

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Simone280802/AskQE_DNLP_2025-2026.git
cd AskQE_DNLP_2025-2026
```

2. Install all required dependencies:
```bash
pip install -r requirements.txt
```

Or install libraries individually:
```bash
pip install transformers==4.36.0
pip install torch==2.1.0
pip install datasets==2.15.0
pip install sentence-transformers==2.2.2
pip install scikit-learn==1.3.2
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install matplotlib==3.8.0
pip install seaborn==0.13.0
```

# Getting Started

The project follows a modular pipeline structure. Each extension builds upon the previous one:

### 1. Baseline Evaluation
Run the baseline AskQE implementation to establish performance benchmarks:
```bash
# BioMQM baseline
jupyter notebook Qwen2.5-3B-Instruct/biomqm/baseline/baseline.ipynb

# ContraTICO baseline
jupyter notebook Qwen2.5-3B-Instruct/contratico/baseline/baseline.ipynb
```

**What it does:**
- Generates questions from source text using Qwen2.5-3B
- Answers questions on both source and translated text
- Establishes baseline performance metrics

### 2. NER-Enhanced Question Generation
Improve question quality by incorporating domain-specific entity recognition:
```bash
# BioMQM with NER
jupyter notebook Qwen2.5-3B-Instruct/biomqm/ner-extension/ner_pipeline.ipynb

# ContraTICO with NER
jupyter notebook Qwen2.5-3B-Instruct/contratico/ner-extension/ner_pipeline.ipynb
```

**What it does:**
- Extracts biomedical entities using `d4data/biomedical-ner-all`
- Generates entity-aware questions targeting specific concepts
- Computes entity-focused accuracy metrics

### 3. Prompt Strategy Ablation
Identify the optimal prompting approach for question answering:
```bash
# BioMQM prompt comparison
jupyter notebook Qwen2.5-3B-Instruct/biomqm/prompt-ablation/prompt_ablation.ipynb

# ContraTICO prompt comparison
jupyter notebook Qwen2.5-3B-Instruct/contratico/prompt-ablation/prompt_ablation.ipynb
```

**What it does:**
- Tests three prompting strategies:
  - **P1-FewShot**: Few-shot learning with 3 examples
  - **P2-CoT**: Chain-of-thought reasoning
  - **P3-Concise**: Minimal instruction prompts
- Evaluates QA accuracy across strategies
- Identifies best-performing approach per dataset

### 4. Metrics Extension
Extend evaluation beyond exact match with semantic metrics:
```bash
# BioMQM advanced metrics
jupyter notebook Qwen2.5-3B-Instruct/biomqm/metrics-extension/metrics-extension.ipynb

# ContraTICO advanced metrics
jupyter notebook Qwen2.5-3B-Instruct/contratico/metrics-extension/metrics-extension.ipynb
```

**What it does:**
- Implements NLI-based agreement using BART-MNLI
- Adds LLM-Judge scoring with Qwen2.5-3B
- Computes semantic similarity with Sentence-BERT
- Performs comprehensive error analysis

---

---

# Research Paper

This implementation extends the work described in:

> **"AskQE: Question Answering as Automatic Evaluation for Machine Translation"**  
> *Dayeon Ki, Kevin Duh, Marine Carpuat* (2025)  
> *University of Maryland, Johns Hopkins University*

**Our Extensions:**
> **"Interpretable Quality Assessment for Machine Translation: A Question-Answering Approach to Semantic Consistency Evaluation"**  
> *Andò S., Baldi F., Bon L., Melchionda L.* (2026)  
> Politecnico di Torino - Deep Natural Language Processing Project

---

## People

**Authors**  
- **Simone Andò** - s346523@studenti.polito.it - Politecnico di Torino, Turin, Italy  
- **Federico Baldi** - s349417@studenti.polito.it - Politecnico di Torino, Turin, Italy  
- **Laura Bon** - s345052@studenti.polito.it - Politecnico di Torino, Turin, Italy  
- **Lorenzo Melchionda** - s339805@studenti.polito.it - Politecnico di Torino, Turin, Italy


**Original Framework**  
- [AskQE](https://github.com/dayeonki/askqe) by Kim et al. (2024)

---

Each notebook is self-contained and handles the entire pipeline for its respective section.
