"""
Confusion Matrix between NLI and LLM-Judge predictions for ContraTICO dataset.
Rows = NLI label, Columns = LLM-Judge label.
Computed per pipeline (atomic, semantic, vanilla).
"""
import json
import os
import csv
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
nli_base = os.path.join(results_dir, "nli")
llm_base = os.path.join(results_dir, "llm-judge")

langs = ["es", "fr", "hi", "tl", "zh"]
pipelines = ["atomic", "semantic", "vanilla"]
label_names = ["entailment", "neutral", "contradiction"]

for pipeline in pipelines:
    nli_dir = os.path.join(nli_base, pipeline)
    llm_dir = os.path.join(llm_base, pipeline)

    # Count matrix[nli_label][llm_label]
    matrix = defaultdict(lambda: defaultdict(int))

    for lang in langs:
        nli_file = os.path.join(nli_dir, f"{lang}-nli.jsonl")
        llm_file = os.path.join(llm_dir, f"{lang}-llm-judge.jsonl")

        if not os.path.exists(nli_file) or not os.path.exists(llm_file):
            print(f"WARNING: Missing file for {pipeline}/{lang}, skipping")
            continue

        with open(nli_file, encoding="utf-8") as fn, open(llm_file, encoding="utf-8") as fl:
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
    out_dir = os.path.join(results_dir, "CONFUSION_MATRIX_LLM-NLI", pipeline)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "confusion_matrix.csv")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["NLI \\ LLM-Judge"] + label_names + ["total"])
        grand_total = 0
        col_totals = {l: 0 for l in label_names}

        for nli_label in label_names:
            row_total = sum(matrix[nli_label][l] for l in label_names)
            row = [nli_label] + [matrix[nli_label][l] for l in label_names] + [row_total]
            w.writerow(row)
            grand_total += row_total
            for l in label_names:
                col_totals[l] += matrix[nli_label][l]

        w.writerow(["total"] + [col_totals[l] for l in label_names] + [grand_total])

    print(f"[{pipeline}] Saved: {out_file}")
    # Print summary
    diag = sum(matrix[l][l] for l in label_names)
    print(f"  Total: {grand_total}, Diagonal (agreement): {diag}, Rate: {round(diag/grand_total*100, 2) if grand_total else 0}%")

print("\nDone!")
