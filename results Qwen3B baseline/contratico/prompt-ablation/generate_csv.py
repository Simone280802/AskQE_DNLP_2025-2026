"""
Generate consolidated metrics CSV from Prompt Ablation evaluation results.
Reads the detailed string_comparison CSVs for each strategy (P1, P2, P3),
averages across configs (vanilla, atomic, semantic) per strategy/lang/perturbation,
and writes a single combined CSV.
"""
import csv
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "prompt_ablation_metrics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATEGIES = ["P1-fewshot", "P2-cot", "P3-concise"]
METRIC_COLS = ["f1", "em", "bleu", "chrf", "sbert"]

rows_out = []

for strategy in STRATEGIES:
    csv_path = os.path.join(EVAL_DIR, strategy, f"string_comparison_{strategy}.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: File not found: {csv_path}")
        continue

    # Accumulate per (lang, perturbation)
    accum = defaultdict(lambda: {"count": 0, "f1": 0, "em": 0, "bleu": 0, "chrf": 0, "sbert": 0})

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["lang"], row["perturbation"])
            n = int(row["count"])
            accum[key]["count"] += 1  # number of configs
            accum[key]["f1"] += float(row["f1"])
            accum[key]["em"] += float(row["em"])
            accum[key]["bleu"] += float(row["bleu"])
            accum[key]["chrf"] += float(row["chrf"])
            accum[key]["sbert"] += float(row["sbert"])

    # Average across configs
    for (lang, perturbation), vals in sorted(accum.items()):
        n_configs = vals["count"]
        rows_out.append({
            "strategy": strategy,
            "lingua": lang,
            "perturbation": perturbation,
            "f1": round(vals["f1"] / n_configs, 4),
            "em": round(vals["em"] / n_configs, 4),
            "chrf": round(vals["chrf"] / n_configs, 4),
            "bleu": round(vals["bleu"] / n_configs, 4),
            "SBERT": round(vals["sbert"] / n_configs, 4),
        })

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["strategy", "lingua", "perturbation", "f1", "em", "chrf", "bleu", "SBERT"])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"CSV generated: {OUTPUT_CSV}")
print(f"Total rows: {len(rows_out)}")
