"""
Generate consolidated metrics CSV from NER extension evaluation results.
Combines string_comparison_results.json and sbert_results.json into a single CSV
grouped by language and perturbation.
"""
import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRING_JSON = os.path.join(BASE_DIR, "evaluation", "string_comparison", "string_comparison_results.json")
SBERT_JSON = os.path.join(BASE_DIR, "evaluation", "sbert", "sbert_results.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ner_extension_metrics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON files
with open(STRING_JSON, "r", encoding="utf-8") as f:
    string_data = json.load(f)

with open(SBERT_JSON, "r", encoding="utf-8") as f:
    sbert_data = json.load(f)

# Extract by_language_perturbation sections
string_lp = string_data["by_language_perturbation"]
sbert_lp = sbert_data["by_language_perturbation"]

# Build rows
rows = []
for key in sorted(string_lp.keys()):
    lang, perturbation = key.split("__", 1)
    s = string_lp[key]
    sbert_val = sbert_lp.get(key, {}).get("avg_similarity", "")
    rows.append({
        "lingua": lang,
        "perturbation": perturbation,
        "f1": round(s["avg_f1"], 4),
        "em": round(s["avg_em"], 4),
        "chrf": round(s["avg_chrf"], 4),
        "bleu": round(s["avg_bleu"], 4),
        "SBERT": round(sbert_val, 4) if isinstance(sbert_val, float) else sbert_val,
    })

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["lingua", "perturbation", "f1", "em", "chrf", "bleu", "SBERT"])
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV generated: {OUTPUT_CSV}")
print(f"Total rows: {len(rows)}")
