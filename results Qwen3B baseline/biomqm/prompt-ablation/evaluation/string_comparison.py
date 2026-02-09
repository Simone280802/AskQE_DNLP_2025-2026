"""
String Comparison Evaluation for Prompt Ablation Study
Adapted from baseline/evaluation/string_comparison/string_comparison.py

Calculates F1, Exact Match, BLEU, chrF between Source Answers and Back-Translation Answers.
Input: Mapped JSONL file (output of mapping step)
Output: JSONL files with scores per language
"""

import json
import os
import argparse
import collections
import string
import re
from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
import numpy as np

# ========================================
# METRIC FUNCTIONS (Inlined from utils.py)
# ========================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if not s:
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def calculate_bleu(prediction, ground_truth):
    try:
        metric = BLEU(effective_order=True)
        # BLEU expects list of references.
        # prediction is string, references is list of strings
        return metric.sentence_score(prediction, [ground_truth]).score
    except:
        return 0.0

def calculate_chrf(prediction, ground_truth):
    try:
        metric = CHRF()
        return metric.sentence_score(prediction, [ground_truth]).score
    except:
        return 0.0

def compare_answers(prediction, ground_truth):
    f1 = f1_score(prediction, ground_truth)
    em = exact_match_score(prediction, ground_truth)
    bleu = calculate_bleu(prediction, ground_truth)
    chrf = calculate_chrf(prediction, ground_truth)
    return f1, em, chrf, bleu

# ========================================
# CONFIGURAZIONE
# ========================================

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

def main():
    parser = argparse.ArgumentParser(description="String Comparison Evaluation for Prompt Ablation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to mapped JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSONL files")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"ERRORE: File mappato non trovato: {args.input_path}")
        return

    print(f"Caricamento dati da: {args.input_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_by_lang = {lang: [] for lang in LANGUAGES}
    
    # Struttura statistiche: stats[lang][severity] = list of (f1, em, chrf, bleu) tuples
    stats = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in LANGUAGES}

    # 1. ELABORAZIONE
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                lang = row.get('lang_tgt')
                
                if lang not in LANGUAGES: 
                    continue

                src_text = row.get('src', '')
                answers_src = row.get('answers_src', [])
                answers_bt = row.get('answers_bt', [])
                
                # Handle single string answer inputs
                if isinstance(answers_src, str): answers_src = [answers_src]
                if isinstance(answers_bt, str): answers_bt = [answers_bt]

                severities = row.get('severities', ["Neutral"])
                if isinstance(severities, str): severities = [severities]
                severity_field = row.get('severity', None)
                if severity_field:
                    severities = [severity_field]

                # Normalizzazione
                pred_list = [str(x) if x else "" for x in answers_bt]
                ref_list = [str(x) if x else "" for x in answers_src]

                # Padding / Troncamento (Compare first answer with first answer usually)
                len_p = len(pred_list)
                len_r = len(ref_list)
                
                if len_r == 0 or len_p == 0:
                   pass
                else: 
                   if len_p < len_r:
                       pred_list.extend([""] * (len_r - len_p))
                   elif len_p > len_r:
                       pred_list = pred_list[:len_r]

                # Calcolo Score
                row_scores = []
                row_f1_sum = 0
                row_em_sum = 0
                row_chrf_sum = 0
                row_bleu_sum = 0
                valid_pairs = 0

                for pred, ref in zip(pred_list, ref_list):
                    if not ref.strip(): continue
                    
                    f1, EM, chrf, bleu = compare_answers(pred, ref)
                    row_scores.append({
                        "f1": f1,
                        "em": EM,
                        "chrf": chrf,
                        "bleu": bleu
                    })
                    
                    row_f1_sum += f1
                    row_em_sum += EM
                    row_chrf_sum += chrf
                    row_bleu_sum += bleu
                    valid_pairs += 1
                
                # Aggiornamento Statistiche (UNWIND)
                if valid_pairs > 0:
                    avg_f1 = row_f1_sum / valid_pairs
                    avg_em = row_em_sum / valid_pairs
                    avg_chrf = row_chrf_sum / valid_pairs
                    avg_bleu = row_bleu_sum / valid_pairs
                    
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[lang][sev].append((avg_f1, avg_em, avg_chrf, avg_bleu))

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
        jsonl_output_file = os.path.join(args.output_dir, f"{lang}.jsonl")
        
        with open(jsonl_output_file, 'w', encoding='utf-8') as out_f:
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Report Unwind
        print(f"\n{'='*50}")
        print(f"String Comparison - Language: {lang}")
        print(f"Total Rows: {len(rows)}")
        print(f"{'='*50}")
        print(f"{'Severity':<10} {'Count':>6} {'F1':>8} {'EM':>8} {'chrF':>8} {'BLEU':>8}")
        print("-" * 55)

        for sev in ALL_SEVERITIES:
            scores_list = stats[lang][sev]
            count = len(scores_list)
            
            if count > 0:
                avg_f1 = sum(s[0] for s in scores_list) / count
                avg_em = sum(s[1] for s in scores_list) / count
                avg_chrf = sum(s[2] for s in scores_list) / count
                avg_bleu = sum(s[3] for s in scores_list) / count
                print(f"{sev:<10} {count:>6} {avg_f1:>8.3f} {avg_em:>8.3f} {avg_chrf:>8.3f} {avg_bleu:>8.3f}")
            else:
                 print(f"{sev:<10} {count:>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        
        # Overall Summary
        all_scores = []
        for sev in ALL_SEVERITIES:
            all_scores.extend(stats[lang][sev])
        
        if all_scores:
            avg_f1 = np.mean([s[0] for s in all_scores])
            avg_em = np.mean([s[1] for s in all_scores])
            avg_chrf = np.mean([s[2] for s in all_scores])
            avg_bleu = np.mean([s[3] for s in all_scores])
            summary_stats.append({
                'lang': lang,
                'avg_f1': avg_f1,
                'avg_em': avg_em,
                'avg_chrf': avg_chrf,
                'avg_bleu': avg_bleu,
                'count': len(all_scores)
            })
        
        print(f"\nSaved: {jsonl_output_file}")
        
    # Save CSV summary
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = os.path.join(os.path.dirname(args.output_dir), "string_comparison_summary_by_lang.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved to: {summary_csv}")

if __name__ == "__main__":
    main()
