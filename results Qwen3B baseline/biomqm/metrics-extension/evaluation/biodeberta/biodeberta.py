"""
BioDeBERTa Semantic Similarity Evaluation - Metrics Extension
Uses pritamdeka/S-BioDeBERTa-snli-mnli for domain-specific biomedical embeddings.

Calculates Cosine Similarity between SOURCE and BACKTRANSLATION answers.
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ========================================
# MODEL SETUP
# ========================================
MODEL_NAME = 'pritamdeka/S-BioDeBERTa-snli-mnli'
tokenizer = None
model = None

def load_model():
    """Lazy load model to avoid loading at import time"""
    global tokenizer, model
    if tokenizer is None:
        print(f"Loading BioDeBERTa model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        print("Model loaded successfully!")

def mean_pooling(model_output, attention_mask):
    """Mean pooling - take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity(pred, ref):
    """Calculate Cosine Similarity between two strings using BioDeBERTa"""
    if not pred or not ref:
        return 0.0
    
    load_model()
    device = next(model.parameters()).device
    
    encoded_pred = tokenizer(pred, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    encoded_ref = tokenizer(ref, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)

    with torch.no_grad():
        pred_output = model(**encoded_pred)
        ref_output = model(**encoded_ref)

    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
    pred_embed = F.normalize(pred_embed, p=2, dim=1)

    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
    ref_embed = F.normalize(ref_embed, p=2, dim=1)

    return F.cosine_similarity(pred_embed, ref_embed, dim=1).item()

# ========================================
# CONFIGURATION
# ========================================
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def main():
    parser = argparse.ArgumentParser(description="BioDeBERTa Evaluation - Metrics Extension")
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
    
    # Statistics structure: stats[lang][severity] = list of cosine_scores
    stats = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in LANGUAGES}

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
                pred_list = [str(x) if x else "" for x in answers_bt]
                ref_list = [str(x) if x else "" for x in answers_src]

                # Padding / Truncation
                len_p = len(pred_list)
                len_r = len(ref_list)
                
                if len_r == 0: 
                    continue
                
                if len_p < len_r:
                    pred_list.extend([""] * (len_r - len_p))
                elif len_p > len_r:
                    pred_list = pred_list[:len_r]

                # Calculate scores
                row_scores = []
                row_sim_sum = 0
                valid_pairs = 0

                for pred, ref in zip(pred_list, ref_list):
                    if not ref.strip(): 
                        continue
                    
                    sim = get_similarity(pred, ref)
                    row_scores.append({"biodeberta_sim": sim})
                    
                    row_sim_sum += sim
                    valid_pairs += 1
                
                # Update statistics (UNWIND by severity)
                if valid_pairs > 0:
                    avg_sim_row = row_sim_sum / valid_pairs
                    
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev].append(avg_sim_row)

                # Output Row
                output_row = {
                    "src": src_text,
                    "severities": severities,
                    "scores": row_scores
                }
                
                results_by_lang[lang].append(output_row)

                # Progress
                if (i + 1) % 100 == 0:
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
            "biodeberta",
            f"{lang}-biodeberta.jsonl"
        )
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)

        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report
        print(f"\n{'='*50}")
        print(f"BioDeBERTa Evaluation - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*50}")
        print(f"{'Severity':<10} {'Count':>6} {'Avg CosSim':>12}")
        print("-" * 30)

        for sev in ALL_SEVERITIES:
            scores_list = stats[lang][sev]
            count = len(scores_list)
            if count > 0:
                avg_val = sum(scores_list) / count
                print(f"{sev:<10} {count:>6} {avg_val:>12.3f}")
            else:
                print(f"{sev:<10} {count:>6} {'N/A':>12}")
        
        print(f"\nSaved: {jsonl_output_file}")


if __name__ == "__main__":
    main()
