"""
String Comparison Evaluation for NER Extension (ContraTICO)

Calculates F1, EM with breakdown by entity type and perturbation type.

Usage:
    python string_comparison_contratico.py --input_dir QA/bt --output_dir evaluation/string_comparison/
"""

import json
import os
import argparse
from utils import compare_answers


LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["alteration", "expansion_impact", "expansion_noimpact",
                  "intensifier", "omission", "spelling", "synonym", "word_order"]


def process_bt_file(filepath):
    """
    Process a single BT JSONL file that already contains answers_src and answers_bt.
    
    Returns list of per-entity metrics:
        [{"entity_type", "f1", "em", "answer_src", "answer_bt", "id", "question"}, ...]
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

            # Match answers by index (same question, same entity)
            for i in range(min(len(answers_src), len(answers_bt))):
                src_answer = answers_src[i].get('answer', '')
                bt_answer = answers_bt[i].get('answer', '')
                entity_type = answers_src[i].get('entity_type', 'unknown')
                question = answers_src[i].get('question', '')

                # Skip if source answer is empty or NOT FOUND
                if not src_answer or src_answer.strip() == '[NOT FOUND]':
                    continue

                f1, em, bleu, chrf = compare_answers(bt_answer, src_answer)

                entity_metrics.append({
                    'id': row.get('id', ''),
                    'entity_type': entity_type,
                    'question': question,
                    'answer_src': src_answer,
                    'answer_bt': bt_answer,
                    'f1': f1,
                    'em': em,
                    'bleu': bleu,
                    'chrf': chrf
                })

    return entity_metrics


def main():
    parser = argparse.ArgumentParser(
        description="String Comparison Evaluation for ContraTICO NER Extension")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to QA/bt directory (contains lang subdirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect all metrics ───────────────────────────────────────
    all_metrics = []              # flat list of per-entity dicts
    per_lang = {}                 # lang -> [metrics]
    per_perturbation = {}         # perturbation -> [metrics]
    per_entity_type = {}          # entity_type -> [metrics]
    per_lang_perturbation = {}    # (lang, perturbation) -> [metrics]

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

            metrics = process_bt_file(filepath)

            # Tag each metric with lang and perturbation
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
    def avg(lst, key):
        if not lst:
            return 0.0
        return sum(m[key] for m in lst) / len(lst)

    # ── Build summary dict ────────────────────────────────────────
    summary = {
        'global': {
            'count': len(all_metrics),
            'avg_f1': avg(all_metrics, 'f1'),
            'avg_em': avg(all_metrics, 'em'),
            'avg_bleu': avg(all_metrics, 'bleu'),
            'avg_chrf': avg(all_metrics, 'chrf'),
        },
        'by_language': {},
        'by_perturbation': {},
        'by_entity_type': {},
        'by_language_perturbation': {},
    }

    for lang, mlist in sorted(per_lang.items()):
        summary['by_language'][lang] = {
            'count': len(mlist),
            'avg_f1': avg(mlist, 'f1'),
            'avg_em': avg(mlist, 'em'),
            'avg_bleu': avg(mlist, 'bleu'),
            'avg_chrf': avg(mlist, 'chrf'),
        }

    for pert, mlist in sorted(per_perturbation.items()):
        summary['by_perturbation'][pert] = {
            'count': len(mlist),
            'avg_f1': avg(mlist, 'f1'),
            'avg_em': avg(mlist, 'em'),
            'avg_bleu': avg(mlist, 'bleu'),
            'avg_chrf': avg(mlist, 'chrf'),
        }

    for et, mlist in sorted(per_entity_type.items()):
        summary['by_entity_type'][et] = {
            'count': len(mlist),
            'avg_f1': avg(mlist, 'f1'),
            'avg_em': avg(mlist, 'em'),
            'avg_bleu': avg(mlist, 'bleu'),
            'avg_chrf': avg(mlist, 'chrf'),
        }

    for (lang, pert), mlist in sorted(per_lang_perturbation.items()):
        key = f"{lang}__{pert}"
        summary['by_language_perturbation'][key] = {
            'count': len(mlist),
            'avg_f1': avg(mlist, 'f1'),
            'avg_em': avg(mlist, 'em'),
            'avg_bleu': avg(mlist, 'bleu'),
            'avg_chrf': avg(mlist, 'chrf'),
        }

    # ── Save summary JSON ─────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "string_comparison_results.json")
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
    print("STRING COMPARISON — GLOBAL")
    print(f"{'='*70}")
    print(f"Total entity comparisons: {summary['global']['count']}")
    print(f"Average F1:   {summary['global']['avg_f1']:.4f}")
    print(f"Average EM:   {summary['global']['avg_em']:.4f}")
    print(f"Average BLEU: {summary['global']['avg_bleu']:.4f}")
    print(f"Average chrF: {summary['global']['avg_chrf']:.4f}")

    print(f"\n{'='*70}")
    print("BY LANGUAGE")
    print(f"{'='*70}")
    print(f"{'Language':<8} {'Count':>8} {'Avg F1':>10} {'Avg EM':>10} {'Avg BLEU':>10} {'Avg chrF':>10}")
    print("-" * 62)
    for lang, stats in sorted(summary['by_language'].items()):
        print(f"{lang:<8} {stats['count']:>8} {stats['avg_f1']:>10.4f} {stats['avg_em']:>10.4f} {stats['avg_bleu']:>10.4f} {stats['avg_chrf']:>10.4f}")

    print(f"\n{'='*70}")
    print("BY PERTURBATION")
    print(f"{'='*70}")
    print(f"{'Perturbation':<22} {'Count':>8} {'Avg F1':>10} {'Avg EM':>10} {'Avg BLEU':>10} {'Avg chrF':>10}")
    print("-" * 76)
    for pert, stats in sorted(summary['by_perturbation'].items()):
        print(f"{pert:<22} {stats['count']:>8} {stats['avg_f1']:>10.4f} {stats['avg_em']:>10.4f} {stats['avg_bleu']:>10.4f} {stats['avg_chrf']:>10.4f}")

    print(f"\n{'='*70}")
    print("BY ENTITY TYPE")
    print(f"{'='*70}")
    print(f"{'Entity Type':<25} {'Count':>8} {'Avg F1':>10} {'Avg EM':>10} {'Avg BLEU':>10} {'Avg chrF':>10}")
    print("-" * 79)
    for et, stats in sorted(summary['by_entity_type'].items()):
        print(f"{et:<25} {stats['count']:>8} {stats['avg_f1']:>10.4f} {stats['avg_em']:>10.4f} {stats['avg_bleu']:>10.4f} {stats['avg_chrf']:>10.4f}")

    print(f"\n{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
