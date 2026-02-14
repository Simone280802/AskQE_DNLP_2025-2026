"""
LLM Judge Evaluation - Metrics Extension
Uses Qwen2.5-3B-Instruct as a judge to classify NLI relationships.

Compares with NLI Classifier (DeBERTa) to assess agreement.
Classifies relationship between SOURCE and BACKTRANSLATION answers as:
- ENTAILMENT: BT answer supports/is consistent with Source answer
- NEUTRAL: Neither clearly supportive nor contradictory
- CONTRADICTION: BT answer contradicts Source answer
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========================================
# MODEL SETUP
# ========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = None
model = None
LABEL_TOKEN_IDS = {}  # Will store map: "ENTAILMENT": id, ...

VALID_LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

# Prompt for NLI classification
JUDGE_PROMPT = """You are a judge evaluating the relationship between two answers to the same question.

Answer A (Source): {answer_src}
Answer B (Backtranslation): {answer_bt}

Classify the relationship as one of:
- ENTAILMENT: Answer B supports or is consistent with Answer A
- NEUTRAL: Answer B is neither clearly supportive nor contradictory  
- CONTRADICTION: Answer B contradicts or is inconsistent with Answer A

Respond with ONLY the label: ENTAILMENT, NEUTRAL, or CONTRADICTION.
Label:"""  # Added "Label:" to guide the model to the next token


def load_model():
    """Lazy load model to avoid loading at import time"""
    global tokenizer, model, LABEL_TOKEN_IDS
    if tokenizer is None:
        print(f"Loading LLM Judge model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print("Model loaded successfully!")
        
        # Identify token IDs for the labels
        # We check both with and without leading space, and Upper/Title case
        # Since prompt ends with "Label:", the next token might have a space or not depending on tokenizer
        # Qwen usually treats " ENTAILMENT" and "ENTAILMENT" differently.
        # We will use the ones that are valid single tokens.
        
        candidates = {
            "ENTAILMENT": [" ENTAILMENT", "ENTAILMENT", " Entailment", "Entailment"],
            "NEUTRAL": [" NEUTRAL", "NEUTRAL", " Neutral", "Neutral"],
            "CONTRADICTION": [" CONTRADICTION", "CONTRADICTION", " Contradiction", "Contradiction"]
        }
        
        print("\nIdentified Token IDs:")
        for label, variants in candidates.items():
            found = False
            for var in variants:
                ids = tokenizer.encode(var, add_special_tokens=False)
                if len(ids) == 1:
                    LABEL_TOKEN_IDS[label] = ids[0]
                    print(f"  {label} ({var}): {ids[0]}")
                    found = True
                    break # Take the first valid single token (prioritizing leading space Uppercase)
            
            if not found:
                print(f"WARNING: Could not find single token for {label}. Using first token of first variant.")
                LABEL_TOKEN_IDS[label] = tokenizer.encode(variants[0], add_special_tokens=False)[0]


def judge_nli(answer_src, answer_bt):
    """
    Use Qwen as judge to classify NLI relationship using Logits.
    Returns: predicted label and probabilities.
    """
    if not answer_src or not answer_bt:
        return "NEUTRAL", {"ENTAILMENT": 0.0, "NEUTRAL": 1.0, "CONTRADICTION": 0.0}
    
    load_model()
    
    prompt = JUDGE_PROMPT.format(answer_src=answer_src, answer_bt=answer_bt)
    
    messages = [
        {"role": "system", "content": "You are a precise judge. Answer only with the label requested."},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        # Get logits of the last token (the one that predicts the next token)
        next_token_logits = outputs.logits[0, -1, :]
    
    # Extract logits for valid labels
    # Order: ENTAILMENT, NEUTRAL, CONTRADICTION
    label_logits = []
    for label in VALID_LABELS:
        tid = LABEL_TOKEN_IDS.get(label)
        if tid is not None:
            label_logits.append(next_token_logits[tid])
        else:
            label_logits.append(torch.tensor(-float('inf')).to(model.device)) # Should not happen

    label_logits_tensor = torch.stack(label_logits)
    
    # Compute Softmax on these 3 values
    probs = F.softmax(label_logits_tensor, dim=0)
    
    # Map back to labels
    probs_dict = {
        label: probs[i].item() for i, label in enumerate(VALID_LABELS)
    }
    
    # Determine the label with max probability
    predicted_label_index = torch.argmax(probs).item()
    predicted_label = VALID_LABELS[predicted_label_index] # e.g. "ENTAILMENT"
    
    return predicted_label, probs_dict


# ========================================
# CONFIGURATION
# ========================================
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Evaluation - Metrics Extension")
    parser.add_argument("--mapped_file_path", type=str, required=True,
                        help="Path to the mapped JSONL file (all-direct-prompting.jsonl)")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base directory for output files")
    args = parser.parse_args()
    
    mapped_file_path = args.mapped_file_path
    output_base_dir = args.output_base_dir

    if not os.path.exists(mapped_file_path):
        print(f"ERROR: Mapped file not found: {mapped_file_path}")
        return

    print(f"Loading data from: {mapped_file_path}")
    
    results_by_lang = {lang: [] for lang in LANGUAGES}
    
    # Statistics: stats[lang][severity][label] = count
    stats = {
        lang: {
            sev: {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0} 
            for sev in ALL_SEVERITIES
        } 
        for lang in LANGUAGES
    }
    total_by_lang = {lang: {sev: 0 for sev in ALL_SEVERITIES} for lang in LANGUAGES}

    # 1. PROCESSING
    with open(mapped_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                lang = row.get('lang_tgt')
                if lang not in LANGUAGES: 
                    continue

                src_text = row.get('src', '')
                answers_src = row.get('answers_src', [])
                answers_bt = row.get('answers_bt', [])
                severities = row.get('severities', ["Neutral"])

                # Normalize
                src_list = [str(x) if x else "" for x in answers_src]
                bt_list = [str(x) if x else "" for x in answers_bt]

                # Padding / Truncation
                len_bt = len(bt_list)
                len_src = len(src_list)
                
                if len_src == 0: 
                    continue
                
                if len_bt < len_src:
                    bt_list.extend([""] * (len_src - len_bt))
                elif len_bt > len_src:
                    bt_list = bt_list[:len_src]

                # Judge each answer pair
                llm_results = []
                for src_ans, bt_ans in zip(src_list, bt_list):
                    if not src_ans.strip(): 
                        continue
                    
                    label, probs = judge_nli(src_ans, bt_ans)
                    llm_results.append({
                        "label": label,
                        "probs": probs
                    })
                    
                    # Update statistics (UNWIND by severity)
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev][label] += 1
                            total_by_lang[lang][sev] += 1

                # Output Row
                output_row = {
                    "src": src_text,
                    "severities": severities,
                    "llm_judge_results": llm_results
                }
                
                results_by_lang[lang].append(output_row)

                # Progress
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} rows...")

            except json.JSONDecodeError:
                print(f"Error at row {i}")
                continue

    # 2. OUTPUT AND REPORT
    for lang in LANGUAGES:
        rows = results_by_lang[lang]
        if not rows:
            print(f"\nNo data for {lang}")
            continue

        # Output file
        jsonl_output_file = os.path.join(
            output_base_dir, 
            "results", 
            "llm-judge",
            f"{lang}-llm-judge.jsonl"
        )
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)

        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report
        print(f"\n{'='*60}")
        print(f"LLM Judge Evaluation - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*60}")
        print(f"{'Severity':<10} {'Entail':>8} {'Neutral':>8} {'Contra':>8} {'Total':>8}")
        print("-" * 50)

        for sev in ALL_SEVERITIES:
            e = stats[lang][sev]["ENTAILMENT"]
            n = stats[lang][sev]["NEUTRAL"]
            c = stats[lang][sev]["CONTRADICTION"]
            total = total_by_lang[lang][sev]
            if total > 0:
                print(f"{sev:<10} {e:>8} {n:>8} {c:>8} {total:>8}")
            else:
                print(f"{sev:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8}")
        
        print(f"\nSaved: {jsonl_output_file}")


if __name__ == "__main__":
    main()
