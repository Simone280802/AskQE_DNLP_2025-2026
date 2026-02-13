"""
Confusion Matrix between NLI and LLM-Judge predictions.
Rows = NLI label, Columns = LLM-Judge label.
"""
import json
import os
import csv
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(os.path.dirname(script_dir), "results")
nli_dir = os.path.join(base, "nli")
llm_dir = os.path.join(base, "llm-judge")

langs = ["de", "es", "fr", "ru", "zh-CN"]
labels = ["entailment", "neutral", "contradiction"]

# Count matrix[nli_label][llm_label]
matrix = defaultdict(lambda: defaultdict(int))

for lang in langs:
    nli_file = os.path.join(nli_dir, f"{lang}-nli.jsonl")
    llm_file = os.path.join(llm_dir, f"{lang}-llm-judge.jsonl")

    with open(nli_file) as fn, open(llm_file) as fl:
        for nli_line, llm_line in zip(fn, fl):
            nli_row = json.loads(nli_line)
            llm_row = json.loads(llm_line)

            nli_results = nli_row.get("nli_results", [])
            llm_results = llm_row.get("llm_judge_results", [])

            n = min(len(nli_results), len(llm_results))
            for i in range(n):
                nli_label = nli_results[i]["label"].lower()
                llm_label = llm_results[i]["label"].lower()
                matrix[nli_label][llm_label] += 1

# Save CSV
out_file = os.path.join(script_dir, "confusion_matrix.csv")

with open(out_file, "w", newline="") as f:
    w = csv.writer(f)
    # Header: NLI \ LLM-Judge, entailment, neutral, contradiction, total
    w.writerow(["NLI \\ LLM-Judge"] + labels + ["total"])
    grand_total = 0
    col_totals = {l: 0 for l in labels}

    for nli_label in labels:
        row_total = sum(matrix[nli_label][l] for l in labels)
        row = [nli_label] + [matrix[nli_label][l] for l in labels] + [row_total]
        w.writerow(row)
        grand_total += row_total
        for l in labels:
            col_totals[l] += matrix[nli_label][l]

    w.writerow(["total"] + [col_totals[l] for l in labels] + [grand_total])

print(f"Saved: {out_file}")

# Print
print(f"\n{'NLI \\ LLM-Judge':<20}", end="")
for l in labels:
    print(f"{l:>16}", end="")
print(f"{'total':>10}")
print("-" * 72)
for nli_label in labels:
    print(f"{nli_label:<20}", end="")
    for llm_label in labels:
        print(f"{matrix[nli_label][llm_label]:>16}", end="")
    print(f"{sum(matrix[nli_label][l] for l in labels):>10}")
print("-" * 72)
print(f"{'total':<20}", end="")
for l in labels:
    print(f"{col_totals[l]:>16}", end="")
print(f"{grand_total:>10}")
