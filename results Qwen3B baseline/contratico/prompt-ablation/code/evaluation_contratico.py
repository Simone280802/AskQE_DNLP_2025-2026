"""
String Comparison Evaluation for Prompt Ablation Study (contraTICO)

Calculates F1, Exact Match, BLEU, chrF between Source Answers and BT Answers.
Reads directly from prompt-ablation QA output files, matching by ID.

Usage:
  python evaluation_contratico.py \
      --prompt_ablation_dir /path/to/prompt-ablation \
      --strategy P1-fewshot
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
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ========================================
# METRIC FUNCTIONS
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
        return metric.sentence_score(prediction, [ground_truth]).score
    except:
        return 0.0

def calculate_chrf(prediction, ground_truth):
    try:
        metric = CHRF()
        return metric.sentence_score(prediction, [ground_truth]).score
    except:
        return 0.0

# ── SBERT ──
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = None
model = None

def load_sbert():
    global tokenizer, model
    if tokenizer is None:
        print(f"Loading SBERT model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_sbert(pred, ref):
    """Calcola Cosine Similarity via SBERT."""
    if not pred or not ref:
        return 0.0
    
    load_sbert()
    device = model.device

    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt').to(device)
    encoded_ref = tokenizer(ref, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        pred_output = model(**encoded_pred)
        ref_output = model(**encoded_ref)

    pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
    pred_embed = F.normalize(pred_embed, p=2, dim=1)

    ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
    ref_embed = F.normalize(ref_embed, p=2, dim=1)

    return F.cosine_similarity(pred_embed, ref_embed, dim=1).item()

def compare_answers(prediction, ground_truth):
    f1 = f1_score(prediction, ground_truth)
    em = exact_match_score(prediction, ground_truth)
    bleu = calculate_bleu(prediction, ground_truth)
    chrf = calculate_chrf(prediction, ground_truth)
    sbert = calculate_sbert(prediction, ground_truth)
    return f1, em, bleu, chrf, sbert

# ========================================
# CONFIGURATION
# ========================================

LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
CONFIGS = ["vanilla", "atomic", "semantic"]
PERTURBATIONS = [
    "alteration", "expansion_impact", "expansion_noimpact",
    "intensifier", "omission", "spelling", "synonym", "word_order",
]


def load_qa_by_id(filepath):
    """Load QA JSONL file and index by ID."""
    by_id = {}
    if not os.path.exists(filepath):
        return by_id
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_id[row.get('id', '')] = row
    return by_id


def main():
    parser = argparse.ArgumentParser(description="Evaluation for Prompt Ablation - contraTICO")
    parser.add_argument("--prompt_ablation_dir", type=str, required=True,
                        help="Path to prompt-ablation directory")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["P1-fewshot", "P2-cot", "P3-concise"],
                        help="Strategy to evaluate")
    args = parser.parse_args()

    strategy = args.strategy
    pa_dir = args.prompt_ablation_dir

    # Output directory for evaluation results
    eval_dir = os.path.join(pa_dir, "evaluation", strategy)
    os.makedirs(eval_dir, exist_ok=True)
    
    load_sbert() # Preload

    all_results = []  # List of dicts for CSV

    for config in CONFIGS:
        # Load source answers for this config
        source_path = os.path.join(pa_dir, "QA", strategy, "source", f"en-{config}.jsonl")
        source_by_id = load_qa_by_id(source_path)

        if not source_by_id:
            print(f"WARNING: No source data for {strategy}/{config}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy} | Config: {config}")
        print(f"Source rows: {len(source_by_id)}")
        print(f"{'=' * 60}")

        for lang in LANGUAGES:
            lang_scores = {pert: [] for pert in PERTURBATIONS}

            for pert in PERTURBATIONS:
                bt_filename = f"{lang}-{config}-{pert}.jsonl"
                bt_path = os.path.join(pa_dir, "QA", strategy, "bt", lang, config, bt_filename)
                bt_by_id = load_qa_by_id(bt_path)

                if not bt_by_id:
                    continue

                # Match source and bt by ID
                for row_id, bt_row in bt_by_id.items():
                    src_row = source_by_id.get(row_id)
                    if not src_row:
                        continue

                    answers_src = src_row.get('answers', [])
                    answers_bt = bt_row.get('answers', [])

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

                    row_f1, row_em, row_bleu, row_chrf, row_sbert = [], [], [], [], []
                    for pred, ref in zip(pred_list, ref_list):
                        if not ref.strip():
                            continue
                        f1, em, bleu, chrf, sbert = compare_answers(pred, ref)
                        row_f1.append(f1)
                        row_em.append(em)
                        row_bleu.append(bleu)
                        row_chrf.append(chrf)
                        row_sbert.append(sbert)

                    if row_f1:
                        lang_scores[pert].append({
                            'f1': np.mean(row_f1),
                            'em': np.mean(row_em),
                            'bleu': np.mean(row_bleu),
                            'chrf': np.mean(row_chrf),
                            'sbert': np.mean(row_sbert),
                        })

            # Print report for this lang
            print(f"\n  Language: {lang}")
            print(f"  {'Perturbation':<22} {'Count':>6} {'F1':>8} {'EM':>8} {'BLEU':>8} {'chrF':>8} {'SBERT':>8}")
            print(f"  {'-' * 72}")

            lang_all_scores = []
            for pert in PERTURBATIONS:
                scores = lang_scores[pert]
                count = len(scores)
                if count > 0:
                    avg_f1 = np.mean([s['f1'] for s in scores])
                    avg_em = np.mean([s['em'] for s in scores])
                    avg_bleu = np.mean([s['bleu'] for s in scores])
                    avg_chrf = np.mean([s['chrf'] for s in scores])
                    avg_sbert = np.mean([s['sbert'] for s in scores])
                    
                    print(f"  {pert:<22} {count:>6} {avg_f1:>8.3f} {avg_em:>8.3f} {avg_bleu:>8.3f} {avg_chrf:>8.3f} {avg_sbert:>8.3f}")
                    lang_all_scores.extend(scores)

                    all_results.append({
                        'strategy': strategy,
                        'config': config,
                        'lang': lang,
                        'perturbation': pert,
                        'count': count,
                        'f1': avg_f1,
                        'em': avg_em,
                        'bleu': avg_bleu,
                        'chrf': avg_chrf,
                        'sbert': avg_sbert,
                    })

            # Lang totals
            if lang_all_scores:
                avg_f1 = np.mean([s['f1'] for s in lang_all_scores])
                avg_em = np.mean([s['em'] for s in lang_all_scores])
                avg_bleu = np.mean([s['bleu'] for s in lang_all_scores])
                avg_chrf = np.mean([s['chrf'] for s in lang_all_scores])
                avg_sbert = np.mean([s['sbert'] for s in lang_all_scores])
                total = len(lang_all_scores)
                print(f"  {'TOTAL':<22} {total:>6} {avg_f1:>8.3f} {avg_em:>8.3f} {avg_bleu:>8.3f} {avg_chrf:>8.3f} {avg_sbert:>8.3f}")

    # Save detailed CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(eval_dir, f"string_comparison_{strategy}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n\nDetailed CSV saved: {csv_path}")

        # Summary by language
        summary = df.groupby('lang')[['f1', 'em', 'bleu', 'chrf', 'sbert']].mean()
        summary_path = os.path.join(eval_dir, f"summary_by_lang_{strategy}.csv")
        summary.to_csv(summary_path)
        print(f"Summary by language: {summary_path}")

        # Summary by config
        summary_cfg = df.groupby('config')[['f1', 'em', 'bleu', 'chrf', 'sbert']].mean()
        summary_cfg_path = os.path.join(eval_dir, f"summary_by_config_{strategy}.csv")
        summary_cfg.to_csv(summary_cfg_path)
        print(f"Summary by config: {summary_cfg_path}")

        # Summary by perturbation
        summary_pert = df.groupby('perturbation')[['f1', 'em', 'bleu', 'chrf', 'sbert']].mean()
        summary_pert_path = os.path.join(eval_dir, f"summary_by_perturbation_{strategy}.csv")
        summary_pert.to_csv(summary_pert_path)
        print(f"Summary by perturbation: {summary_pert_path}")

        # Overall
        print(f"\n{'=' * 60}")
        print(f"OVERALL ({strategy})")
        print(f"{'=' * 60}")
        print(f"  F1:   {df['f1'].mean():.4f}")
        print(f"  EM:   {df['em'].mean():.4f}")
        print(f"  BLEU: {df['bleu'].mean():.4f}")
        print(f"  chrF: {df['chrf'].mean():.4f}")
        print(f"  SBERT:{df['sbert'].mean():.4f}")



if __name__ == "__main__":
    main()
