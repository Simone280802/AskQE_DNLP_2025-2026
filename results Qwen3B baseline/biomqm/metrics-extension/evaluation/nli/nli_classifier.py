"""
NLI Classifier Evaluation - Metrics Extension
Uses microsoft/deberta-v3-large-mnli for Natural Language Inference.

Classifies relationship between SOURCE and BACKTRANSLATION answers as:
- entailment: BT answer supports/is consistent with Source answer
- neutral: Neither clearly supportive nor contradictory
- contradiction: BT answer contradicts Source answer
"""

import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================================
# MODEL SETUP
# ========================================
MODEL_NAME = 'microsoft/deberta-v3-large-mnli'
LABELS = ['contradiction', 'neutral', 'entailment']  # Model's label order
tokenizer = None
model = None

def load_model():
    """Lazy load model to avoid loading at import time"""
    global tokenizer, model
    if tokenizer is None:
        print(f"Loading NLI model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        print("Model loaded successfully!")

def classify_nli(premise, hypothesis):
    """
    Classify NLI relationship between premise (source answer) and hypothesis (BT answer).
    Returns: (label, probabilities_dict)
    """
    if not premise or not hypothesis:
        return "neutral", {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
    
    load_model()
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors='pt', 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    
    # Map to label names
    probs_dict = {label: round(prob, 4) for label, prob in zip(LABELS, probs)}
    predicted_label = LABELS[torch.argmax(logits, dim=1).item()]
    
    return predicted_label, probs_dict

# ========================================
# CONFIGURATION
# ========================================
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def main():
    parser = argparse.ArgumentParser(description="NLI Classifier Evaluation - Metrics Extension")
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
            sev: {"entailment": 0, "neutral": 0, "contradiction": 0} 
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

                # Classify NLI for each answer pair
                nli_results = []
                for src_ans, bt_ans in zip(src_list, bt_list):
                    if not src_ans.strip(): 
                        continue
                    
                    label, probs = classify_nli(src_ans, bt_ans)
                    nli_results.append({
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
                    "nli_results": nli_results
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
            "nli",
            f"{lang}-nli.jsonl"
        )
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)

        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report
        print(f"\n{'='*60}")
        print(f"NLI Classifier Evaluation - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*60}")
        print(f"{'Severity':<10} {'Entail':>8} {'Neutral':>8} {'Contra':>8} {'Total':>8}")
        print("-" * 50)

        for sev in ALL_SEVERITIES:
            e = stats[lang][sev]["entailment"]
            n = stats[lang][sev]["neutral"]
            c = stats[lang][sev]["contradiction"]
            total = total_by_lang[lang][sev]
            if total > 0:
                print(f"{sev:<10} {e:>8} {n:>8} {c:>8} {total:>8}")
            else:
                print(f"{sev:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8}")
        
        print(f"\nSaved: {jsonl_output_file}")


if __name__ == "__main__":
    main()
