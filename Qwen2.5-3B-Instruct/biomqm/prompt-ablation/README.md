# Extension 2: Prompt Ablation Study

This extension investigates how **prompting strategy** affects QA accuracy in the AskQE framework. Three strategies are compared against the baseline:

| Strategy | Description |
|----------|-------------|
| **P1 — Few-Shot** | Includes example QA pairs in the prompt |
| **P2 — Chain-of-Thought** | Asks the model to reason step-by-step before answering |
| **P3 — Concise** | Instructs the model to give short, direct answers |

## Pipeline Overview

```
Baseline QG output (qwen-3b.jsonl)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
┌────────────────────┐   ┌────────────────────────────┐
│ 1. Source QA       │   │ 2-6. BT QA (×5 languages)  │
│   × 3 strategies  │   │   × 3 strategies            │
└────────┬───────────┘   └──────────┬─────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
         ┌─────────────────┐
         │ 7. Mapping       │  Combine source + BT answers
         │    + Evaluation  │  SBERT + String Comparison
         └─────────────────┘
```


```
prompt-ablation/
├── README.md                         ← You are here
├── prompt_ablation.ipynb             # Main notebook (run this)
├── clean_cot_answers.py              # CoT answer cleaning script
├── mapping.py                        # Mapping module
├── code/
│   ├── prompts.py                    # Prompt templates (P1/P2/P3)
│   └── qa_ablation.py                # QA script with strategy support
├── QA/                               # QA results per strategy
│   ├── P1-fewshot/
│   ├── P2-cot/
│   └── P3-concise/
└── evaluation/                       # Evaluation SBERT + String Comparison
```

## Pipeline Steps

### 1. Source QA
We generate answers for the source text using all three strategies (P1, P2, P3).
- **Script**: `code/qa_ablation.py --mode source`
- **Output**: `QA/{strategy}/source-{strategy}.jsonl`

### 2. Backtranslation QA
We generate answers for the backtranslated text across all languages and strategies.
- **Script**: `code/qa_ablation.py --mode bt`
- **Output**: `QA/{strategy}/bt-{lang}-{strategy}.jsonl`

### 3. CoT Cleaning (Specific to P2)
Chain-of-Thought (P2) produces verbose reasoning steps. We run a cleaning script to extract the final answer before evaluation.
- **Script**: `clean_cot_answers.py`
- **Output**: `QA/P2-cot/clean/clean-*.jsonl`

### 4. Mapping & Evaluation
We map the answers (using cleaned versions where applicable) and compute metrics.
- **Scripts**: `mapping.py`, `evaluation/sbert.py`, `evaluation/string_comparison.py`

## Usage

The entire pipeline is encapsulated in a single notebook:

**[prompt_ablation.ipynb](prompt_ablation.ipynb)**

Open this notebook in Jupyter, Google Colab, or Kaggle. It will:
1. Setup the environment and download models
2. Run Source QA for all strategies
3. Run BT QA for all languages and strategies
4. Clean CoT answers automatically
5. Map and evaluate all results

### ContraTICO Variant

The same pipeline is available for the **ContraTICO** dataset (synthetic errors) in:
- `../../contratico/prompt-ablation/prompt_ablation.ipynb`

It follows the same logic but uses ContraTICO's specific languages and perturbation types.

