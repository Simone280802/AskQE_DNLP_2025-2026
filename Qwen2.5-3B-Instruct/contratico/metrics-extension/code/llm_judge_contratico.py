"""
LLM Judge Evaluation for contraTICO
Uses Qwen2.5-3B-Instruct as NLI judge to classify entailment/neutral/contradiction.

Reads QA source + bt files directly from the baseline directory.

Usage:
    python llm_judge_contratico.py \
        --config vanilla \
        --baseline_dir /path/to/contratico/baseline \
        --output_dir /path/to/contratico/metrics-extension \
        [--max_rows 42]
"""

import json
import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========================================
# NLI PROMPT
# ========================================
NLI_PROMPT = """Task: Classify the relationship between Answer A (source) and Answer B (backtranslation).

Answer A (Source): {answer_src}
Answer B (Backtranslation): {answer_bt}

Classify the relationship as one of:
- ENTAILMENT: Answer B supports or is consistent with Answer A
- NEUTRAL: Answer B is neither clearly supportive nor contradictory  
- CONTRADICTION: Answer B contradicts or is inconsistent with Answer A

Respond with ONLY the label: ENTAILMENT, NEUTRAL, or CONTRADICTION.
Label:"""

# ========================================
# MODEL SETUP
# ========================================
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = None
model = None

LABEL_TOKENS = None


def load_model():
    """Lazy load model."""
    global tokenizer, model, LABEL_TOKENS
    if tokenizer is None:
        print(f"Loading LLM Judge model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()

        # Pre-compute label token IDs
        LABEL_TOKENS = {
            "ENTAILMENT": tokenizer.encode("ENTAILMENT", add_special_tokens=False)[0],
            "NEUTRAL": tokenizer.encode("NEUTRAL", add_special_tokens=False)[0],
            "CONTRADICTION": tokenizer.encode("CONTRADICTION", add_special_tokens=False)[0],
        }
        print("Model loaded!")


def judge_nli(answer_src, answer_bt):
    """Use Qwen as judge to classify NLI. Returns: (label, probs_dict)."""
    if not answer_src or not answer_bt:
        return "NEUTRAL", {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0}

    load_model()

    prompt = NLI_PROMPT.format(answer_src=answer_src, answer_bt=answer_bt)
    messages = [
        {"role": "system", "content": "You are a precise NLI classifier."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

    # Extract logits for the 3 labels
    label_logits = torch.tensor([
        logits[LABEL_TOKENS["ENTAILMENT"]].item(),
        logits[LABEL_TOKENS["NEUTRAL"]].item(),
        logits[LABEL_TOKENS["CONTRADICTION"]].item(),
    ])

    probs = torch.softmax(label_logits, dim=0).tolist()
    labels = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    probs_dict = {l: round(p, 4) for l, p in zip(labels, probs)}
    predicted = labels[torch.argmax(label_logits).item()]

    return predicted, probs_dict


# ========================================
# CONFIGURATION
# ========================================
LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = [
    "alteration", "expansion_impact", "expansion_noimpact",
    "intensifier", "omission", "spelling", "synonym", "word_order",
]


def main():
    parser = argparse.ArgumentParser(description="LLM Judge for contraTICO")
    parser.add_argument("--config", type=str, required=True,
                        choices=["vanilla", "atomic", "semantic"],
                        help="QG configuration to evaluate")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Path to contraTICO baseline directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_rows", type=int, default=42,
                        help="Max rows per file to process (default: 42)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    config = args.config
    baseline_dir = args.baseline_dir
    max_rows = args.max_rows

    # ── Load QA source answers ──
    source_path = os.path.join(baseline_dir, "QA", "source", f"en-{config}.jsonl")
    if not os.path.exists(source_path):
        print(f"ERROR: Source file not found: {source_path}")
        return

    source_data = []
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                source_data.append(json.loads(line))
    # Random sample with seed
    if len(source_data) > max_rows:
        rng = random.Random(args.seed)
        source_data = rng.sample(source_data, max_rows)
    print(f"Sampled {len(source_data)} source rows from {source_path} (seed={args.seed})")

    # Index source answers by id
    source_by_id = {row['id']: row for row in source_data}

    # ── Aggregate statistics ──
    stats = {
        lang: {
            pert: {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
            for pert in PERTURBATIONS
        }
        for lang in LANGUAGES
    }
    total_by_lang = {lang: {pert: 0 for pert in PERTURBATIONS} for lang in LANGUAGES}
    results_by_lang = {lang: [] for lang in LANGUAGES}

    # ── Process each lang × perturbation ──
    total_processed = 0
    for lang in LANGUAGES:
        for pert in PERTURBATIONS:
            bt_filename = f"{lang}-{config}-{pert}.jsonl"
            bt_path = os.path.join(baseline_dir, "QA", "bt", lang, config, bt_filename)

            if not os.path.exists(bt_path):
                print(f"SKIP: {bt_path} not found")
                continue

            # Load bt answers
            bt_data = []
            with open(bt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bt_data.append(json.loads(line))
            # No slicing needed - id matching filters to sampled source rows

            # Pair source + bt by id
            for bt_row in bt_data:
                row_id = bt_row.get('id', '')
                src_row = source_by_id.get(row_id)
                if not src_row:
                    continue

                answers_src = src_row.get('answers', [])
                answers_bt = bt_row.get('answers', [])

                src_list = [str(x) if x else "" for x in answers_src]
                bt_list = [str(x) if x else "" for x in answers_bt]

                if len(src_list) == 0:
                    continue

                # Pad/truncate bt to match src length
                if len(bt_list) < len(src_list):
                    bt_list.extend([""] * (len(src_list) - len(bt_list)))
                elif len(bt_list) > len(src_list):
                    bt_list = bt_list[:len(src_list)]

                # Judge each pair
                llm_results = []
                for src_ans, bt_ans in zip(src_list, bt_list):
                    if not src_ans.strip():
                        continue

                    label, probs = judge_nli(src_ans, bt_ans)
                    llm_results.append({"label": label, "probs": probs})

                    stats[lang][pert][label] += 1
                    total_by_lang[lang][pert] += 1

                output_row = {
                    "id": row_id,
                    "src": src_row.get('en', ''),
                    "perturbation": pert,
                    "llm_judge_results": llm_results,
                }
                results_by_lang[lang].append(output_row)
                total_processed += 1

            print(f"[{total_processed}] Processed {lang}/{config}/{pert}")

    # ── Save results ──
    for lang in LANGUAGES:
        rows = results_by_lang[lang]
        if not rows:
            continue

        out_path = os.path.join(
            args.output_dir, "results", "llm-judge", config, f"{lang}-llm-judge.jsonl"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'w', encoding='utf-8') as f_out:
            for r in rows:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report
        print(f"\n{'=' * 60}")
        print(f"LLM Judge - {lang} ({config})")
        print(f"Total Rows: {len(rows)}")
        print(f"{'=' * 60}")
        print(f"{'Perturbation':<22} {'Entail':>8} {'Neutral':>8} {'Contra':>8} {'Total':>8}")
        print("-" * 58)

        for pert in PERTURBATIONS:
            e = stats[lang][pert]["ENTAILMENT"]
            n = stats[lang][pert]["NEUTRAL"]
            c = stats[lang][pert]["CONTRADICTION"]
            total = total_by_lang[lang][pert]
            if total > 0:
                print(f"{pert:<22} {e:>8} {n:>8} {c:>8} {total:>8}")

        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
