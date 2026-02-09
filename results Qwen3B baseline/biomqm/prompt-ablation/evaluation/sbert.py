"""
SBERT Semantic Similarity Evaluation for Prompt Ablation Study
Adapted from baseline/evaluation/sbert/sbert.py

Calculates Cosine Similarity using SBERT between Source Answers and Back-Translation Answers.
Input: Mapped JSONL file (output of mapping step)
Output: JSONL files with scores per language
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

# ========================================
# SBERT MODEL SETUP
# ========================================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# Load model globally to avoid reloading for every row
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity(pred, ref):
    """Calcola Cosine Similarity tra due stringhe"""
    if not pred or not ref:
        return 0.0
        
    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        pred_output = model(**encoded_pred)
        ref_output = model(**encoded_ref)

    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
    pred_embed = F.normalize(pred_embed, p=2, dim=1)

    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
    ref_embed = F.normalize(ref_embed, p=2, dim=1)

    return F.cosine_similarity(pred_embed, ref_embed, dim=1).item()

# ========================================
# CONFIGURAZIONE
# ========================================

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

def main():
    parser = argparse.ArgumentParser(description="SBERT Evaluation for Prompt Ablation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to mapped JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSONL files")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"ERRORE: File mappato non trovato: {args.input_path}")
        return

    print(f"Caricamento dati da: {args.input_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_by_lang = {lang: [] for lang in LANGUAGES}
    
    # Struttura statistiche: stats[lang][severity] = list of cosine_scores
    stats = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in LANGUAGES}

    # 1. ELABORAZIONE
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                lang = row.get('lang_tgt')
                
                # Check if lang is in our list (if not, maybe skip or log)
                if lang not in LANGUAGES: 
                    # If language field is missing or different, skip
                    continue

                src_text = row.get('src', '')
                answers_src = row.get('answers_src', [])
                answers_bt = row.get('answers_bt', [])
                # Handle single string answer inputs (just in case)
                if isinstance(answers_src, str): answers_src = [answers_src]
                if isinstance(answers_bt, str): answers_bt = [answers_bt]

                severities = row.get('severities', ["Neutral"])
                # If severities is string (e.g. "Critical"), wrap in list
                if isinstance(severities, str): severities = [severities]
                # In prompt ablation mapping, 'severity' might be a single string field
                severity_field = row.get('severity', None)
                if severity_field:
                    severities = [severity_field]

                # Normalizzazione
                pred_list = [str(x) if x else "" for x in answers_bt]
                ref_list = [str(x) if x else "" for x in answers_src]

                # Padding / Troncamento (Compare first answer with first answer usually, or all pairs)
                # Baseline logic extended lists to match length. 
                # For prompt ablation we often have 1 answer vs 1 answer.
                len_p = len(pred_list)
                len_r = len(ref_list)
                
                if len_r == 0 or len_p == 0: 
                    # Empty answers case
                    pass
                else:
                    if len_p < len_r:
                        pred_list.extend([""] * (len_r - len_p))
                    elif len_p > len_r:
                        pred_list = pred_list[:len_r]

                # Calcolo Score
                row_scores = []
                row_sim_sum = 0
                valid_pairs = 0

                # If both lists empty or one empty, we might have 0 score.
                # Logic above handles padding, so check zip
                for pred, ref in zip(pred_list, ref_list):
                    # Skip if ref is empty? Baseline says: "if not ref.strip(): continue"
                    if not ref.strip(): continue
                    
                    sim = get_similarity(pred, ref)
                    row_scores.append({"sbert_sim": sim})
                    
                    row_sim_sum += sim
                    valid_pairs += 1
                
                # Aggiornamento Statistiche (UNWIND)
                if valid_pairs > 0:
                    avg_sim_row = row_sim_sum / valid_pairs
                    
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev].append(avg_sim_row)
                else:
                     # Handle case with no valid pairs (e.g. empty answers)
                     # Assign 0.0? Or just skip logic?
                     # Baseline logic relies on valid_pairs > 0 to add to stats.
                     pass

                # Output Row
                output_row = {
                    "src": src_text,
                    "severities": severities,
                    "scores": row_scores,
                    "answers_src": answers_src,
                    "answers_bt": answers_bt,
                    "strategy": row.get('strategy', 'unknown')
                }
                
                results_by_lang[lang].append(output_row)

            except json.JSONDecodeError:
                print(f"Errore riga {i}")
                continue

    # 2. OUTPUT E REPORT
    summary_stats = []

    for lang in LANGUAGES:
        rows = results_by_lang[lang]
        if not rows:
            print(f"\nNessun dato per {lang}")
            continue

        # File output
        # E.g. output_dir/de.jsonl
        jsonl_output_file = os.path.join(args.output_dir, f"{lang}.jsonl")
        
        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report Unwind
        print(f"\n{'='*50}")
        print(f"SBERT Evaluation - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*50}")
        print(f"{'Severity':<10} {'Count':>6} {'Avg CosSim':>12}")
        print("-" * 30)

        lang_total_sum = 0
        lang_total_count = 0

        for sev in ALL_SEVERITIES:
            scores_list = stats[lang][sev]
            count = len(scores_list)
            if count > 0:
                avg_val = sum(scores_list) / count
                print(f"{sev:<10} {count:>6} {avg_val:>12.3f}")
                lang_total_sum += sum(scores_list)
                lang_total_count += count
            else:
                print(f"{sev:<10} {count:>6} {'N/A':>12}")
        
        # Calculate overall average for this language (weighted by severity occurrences or raw?)
        # Since one row can have multiple severities, average of averages might be biased.
        # But stats[lang][sev] contains the raw scores.
        all_scores = []
        for sev in ALL_SEVERITIES:
            all_scores.extend(stats[lang][sev])
        
        overall_avg = np.mean(all_scores) if all_scores else 0.0
        summary_stats.append({
            'lang': lang, 
            'avg_similarity': overall_avg,
            'count': len(all_scores)
        })
        
        print(f"\nSaved: {jsonl_output_file}")
    
    # Save CSV summary
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = os.path.join(os.path.dirname(args.output_dir), "sbert_summary_by_lang.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved to: {summary_csv}")

if __name__ == "__main__":
    main()
