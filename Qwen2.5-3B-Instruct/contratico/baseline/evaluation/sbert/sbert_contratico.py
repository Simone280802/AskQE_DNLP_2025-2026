"""
SBERT Evaluation for ContraTICO Baseline

Computes cosine similarity between source and BT answers using SBERT.

Usage:
    python sbert_contratico.py --base_dir /path/to/baseline
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["alteration", "omission"]
PIPELINES = ["vanilla", "atomic", "semantic"]

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_similarity(tokenizer, model, pred, ref):
    """Compute cosine similarity between two strings."""
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


def main():
    parser = argparse.ArgumentParser(description="SBERT Evaluation for ContraTICO Baseline")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Baseline directory containing mapping/ folder")
    args = parser.parse_args()

    mapping_dir = os.path.join(args.base_dir, "mapping")
    output_dir = os.path.join(args.base_dir, "evaluation", "sbert")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(mapping_dir):
        print(f"ERROR: Mapping directory not found: {mapping_dir}")
        print("Run mapping_contratico.py first!")
        return

    # Load SBERT model
    print(f"Loading SBERT model: {SBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL)
    model = AutoModel.from_pretrained(SBERT_MODEL)
    model.eval()
    print("Model loaded!")

    for pipeline in PIPELINES:
        print(f"\n{'='*60}")
        print(f"SBERT Evaluation - Pipeline: {pipeline}")
        print(f"{'='*60}")

        for lang in LANGUAGES:
            total_sim = 0
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
                            sim = get_similarity(tokenizer, model, pred, ref)
                            total_sim += sim
                            total_count += 1

            if total_count > 0:
                avg_sim = total_sim / total_count
                print(f"  {lang}: SBERT={avg_sim:.3f}  ({total_count} pairs)")

                # Save per-language results
                result = {"lang": lang, "pipeline": pipeline,
                          "avg_sbert": avg_sim, "count": total_count}
                out_file = os.path.join(output_dir, f"{lang}-{pipeline}.jsonl")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                print(f"  {lang}: No data")

    print(f"\n{'='*60}")
    print("SBERT evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
