"""
SBERT Semantic Similarity Evaluation for NER Extension (ContraTICO)

Calculates cosine similarity with breakdown by entity type and perturbation type.

Usage:
    python sbert_contratico.py --input_dir QA/bt --output_dir evaluation/sbert/
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["alteration", "expansion_impact", "expansion_noimpact",
                  "intensifier", "omission", "spelling", "synonym", "word_order"]


def load_sbert_model():
    """Load SBERT model and tokenizer."""
    print(f"Loading SBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_similarity(tokenizer, model, text1, text2):
    """Calculate cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    encoded1 = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
    encoded2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        output1 = model(**encoded1)
        output2 = model(**encoded2)

    embed1 = mean_pooling(output1, encoded1['attention_mask'])
    embed1 = F.normalize(embed1, p=2, dim=1)

    embed2 = mean_pooling(output2, encoded2['attention_mask'])
    embed2 = F.normalize(embed2, p=2, dim=1)

    return F.cosine_similarity(embed1, embed2, dim=1).item()


def process_bt_file(filepath, tokenizer, model):
    """
    Process a single BT JSONL file that already contains answers_src and answers_bt.

    Returns list of per-entity metrics:
        [{"entity_type", "similarity", "answer_src", "answer_bt", "id", "question"}, ...]
    """
    entity_metrics = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            answers_src = row.get('answers_src', [])
            answers_bt = row.get('answers_bt', [])

            for i in range(min(len(answers_src), len(answers_bt))):
                src_answer = answers_src[i].get('answer', '')
                bt_answer = answers_bt[i].get('answer', '')
                entity_type = answers_src[i].get('entity_type', 'unknown')
                question = answers_src[i].get('question', '')

                # Skip if source answer is empty or NOT FOUND
                if not src_answer or src_answer.strip() == '[NOT FOUND]':
                    continue

                # Skip if BT answer is also NOT FOUND (similarity = 0 otherwise dominates)
                if bt_answer.strip() == '[NOT FOUND]':
                    sim = 0.0
                else:
                    sim = get_similarity(tokenizer, model, src_answer, bt_answer)

                entity_metrics.append({
                    'id': row.get('id', ''),
                    'entity_type': entity_type,
                    'question': question,
                    'answer_src': src_answer,
                    'answer_bt': bt_answer,
                    'similarity': sim,
                })

    return entity_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SBERT Evaluation for ContraTICO NER Extension")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to QA/bt directory (contains lang subdirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    tokenizer, model = load_sbert_model()

    # ── Collect all metrics ───────────────────────────────────────
    all_metrics = []
    per_lang = {}
    per_perturbation = {}
    per_entity_type = {}
    per_lang_perturbation = {}

    for lang in LANGUAGES:
        lang_dir = os.path.join(args.input_dir, lang)
        if not os.path.isdir(lang_dir):
            print(f"  Skipping language {lang} (dir not found)")
            continue

        for pert in PERTURBATIONS:
            filepath = os.path.join(lang_dir, f"{pert}.jsonl")
            if not os.path.isfile(filepath):
                print(f"  Skipping {lang}/{pert}.jsonl (file not found)")
                continue

            print(f"  Processing {lang}/{pert}.jsonl ...")
            metrics = process_bt_file(filepath, tokenizer, model)

            for m in metrics:
                m['lang'] = lang
                m['perturbation'] = pert

            all_metrics.extend(metrics)

            per_lang.setdefault(lang, []).extend(metrics)
            per_perturbation.setdefault(pert, []).extend(metrics)
            per_lang_perturbation.setdefault((lang, pert), []).extend(metrics)

            for m in metrics:
                per_entity_type.setdefault(m['entity_type'], []).extend([m])

    if not all_metrics:
        print("No metrics found. Check input_dir structure.")
        return

    # ── Helper ────────────────────────────────────────────────────
    def avg(lst, key='similarity'):
        if not lst:
            return 0.0
        return sum(m[key] for m in lst) / len(lst)

    # ── Build summary dict ────────────────────────────────────────
    summary = {
        'global': {
            'count': len(all_metrics),
            'avg_similarity': avg(all_metrics),
        },
        'by_language': {},
        'by_perturbation': {},
        'by_entity_type': {},
        'by_language_perturbation': {},
    }

    for lang, mlist in sorted(per_lang.items()):
        summary['by_language'][lang] = {
            'count': len(mlist),
            'avg_similarity': avg(mlist),
        }

    for pert, mlist in sorted(per_perturbation.items()):
        summary['by_perturbation'][pert] = {
            'count': len(mlist),
            'avg_similarity': avg(mlist),
        }

    for et, mlist in sorted(per_entity_type.items()):
        summary['by_entity_type'][et] = {
            'count': len(mlist),
            'avg_similarity': avg(mlist),
        }

    for (lang, pert), mlist in sorted(per_lang_perturbation.items()):
        key = f"{lang}__{pert}"
        summary['by_language_perturbation'][key] = {
            'count': len(mlist),
            'avg_similarity': avg(mlist),
        }

    # ── Save summary JSON ─────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "sbert_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")

    # ── Save per-language detailed JSONL ───────────────────────────
    for lang, mlist in per_lang.items():
        lang_path = os.path.join(args.output_dir, f"{lang}.jsonl")
        with open(lang_path, 'w', encoding='utf-8') as f:
            for m in mlist:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    # ── Print summary to console ──────────────────────────────────
    print(f"\n{'='*70}")
    print("SBERT — GLOBAL")
    print(f"{'='*70}")
    print(f"Total entity comparisons: {summary['global']['count']}")
    print(f"Average Similarity: {summary['global']['avg_similarity']:.4f}")

    print(f"\n{'='*70}")
    print("BY LANGUAGE")
    print(f"{'='*70}")
    print(f"{'Language':<8} {'Count':>8} {'Avg Similarity':>15}")
    print("-" * 33)
    for lang, stats in sorted(summary['by_language'].items()):
        print(f"{lang:<8} {stats['count']:>8} {stats['avg_similarity']:>15.4f}")

    print(f"\n{'='*70}")
    print("BY PERTURBATION")
    print(f"{'='*70}")
    print(f"{'Perturbation':<22} {'Count':>8} {'Avg Similarity':>15}")
    print("-" * 47)
    for pert, stats in sorted(summary['by_perturbation'].items()):
        print(f"{pert:<22} {stats['count']:>8} {stats['avg_similarity']:>15.4f}")

    print(f"\n{'='*70}")
    print("BY ENTITY TYPE")
    print(f"{'='*70}")
    print(f"{'Entity Type':<25} {'Count':>8} {'Avg Similarity':>15}")
    print("-" * 50)
    for et, stats in sorted(summary['by_entity_type'].items()):
        print(f"{et:<25} {stats['count']:>8} {stats['avg_similarity']:>15.4f}")

    print(f"\n{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
