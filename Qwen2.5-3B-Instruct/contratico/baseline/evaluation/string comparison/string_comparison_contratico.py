"""
String Comparison Evaluation for ContraTICO Baseline

Calculates F1, EM, chrF, BLEU between source and BT answers from mapped files.

Usage:
    python string_comparison_contratico.py --base_dir /path/to/baseline
"""

import json
import os
import sys
import argparse

# Add parent dirs to path for utils
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try to find utils in the NER extension code dir
ner_code = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "ner-extension", "code")
if os.path.exists(ner_code):
    sys.path.insert(0, ner_code)

# Also check BioMQM for utils
biomqm_eval = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))),
                           "biomqm", "baseline", "evaluation", "string comparison")
if os.path.exists(biomqm_eval):
    sys.path.insert(0, biomqm_eval)

try:
    from utils import compare_answers
except ImportError:
    # Fallback: inline basic F1/EM
    import nltk
    nltk.download("punkt", quiet=True)
    from collections import Counter

    def _tokenize(text):
        return nltk.word_tokenize(text.lower())

    def compare_answers(pred, ref):
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            return 0.0, 0.0, 0.0, 0.0
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        em = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
        return f1, em, 0.0, 0.0


LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["alteration", "omission"]
PIPELINES = ["vanilla", "atomic", "semantic"]


def main():
    parser = argparse.ArgumentParser(description="String Comparison for ContraTICO Baseline")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Baseline directory containing mapping/ folder")
    args = parser.parse_args()

    mapping_dir = os.path.join(args.base_dir, "mapping")
    output_dir = os.path.join(args.base_dir, "evaluation", "string comparison")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(mapping_dir):
        print(f"ERROR: Mapping directory not found: {mapping_dir}")
        print("Run mapping_contratico.py first!")
        return

    for pipeline in PIPELINES:
        print(f"\n{'='*60}")
        print(f"String Comparison - Pipeline: {pipeline}")
        print(f"{'='*60}")

        for lang in LANGUAGES:
            total_f1 = 0
            total_em = 0
            total_count = 0

            for pert in PERTURBATIONS:
                mapped_file = os.path.join(mapping_dir, f"{lang}-{pipeline}-{pert}.jsonl")
                if not os.path.exists(mapped_file):
                    continue

                with open(mapped_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            row = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue

                        answers_src = row.get("answers_src", [])
                        answers_bt = row.get("answers_bt", [])

                        n = min(len(answers_src), len(answers_bt))
                        for i in range(n):
                            ref = str(answers_src[i]) if answers_src[i] else ""
                            pred = str(answers_bt[i]) if answers_bt[i] else ""
                            if not ref.strip():
                                continue
                            f1, em, chrf, bleu = compare_answers(pred, ref)
                            total_f1 += f1
                            total_em += em
                            total_count += 1

            if total_count > 0:
                avg_f1 = total_f1 / total_count
                avg_em = total_em / total_count
                print(f"  {lang}: F1={avg_f1:.3f}  EM={avg_em:.3f}  ({total_count} pairs)")

                # Save per-language results
                result = {"lang": lang, "pipeline": pipeline,
                          "avg_f1": avg_f1, "avg_em": avg_em, "count": total_count}
                out_file = os.path.join(output_dir, f"{lang}-{pipeline}.jsonl")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                print(f"  {lang}: No data")

    print(f"\n{'='*60}")
    print("String comparison complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
