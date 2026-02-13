"""
Confidence analysis for NLI and LLM-Judge on ContraTICO dataset.
Per pipeline (atomic, semantic, vanilla):
  1. Average confidence when assigning each label
  2. Average confidence when both models agree on the same label
"""
import json
import os
import csv
import numpy as np
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

    # Confidence when assigning
    conf_assigned = {
        "nli": defaultdict(list),
        "llm": defaultdict(list),
    }

    # Confidence when both agree
    conf_agree = {label: {"nli": [], "llm": []} for label in label_names}

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

                    # NLI confidence
                    nli_probs = nli_results[i].get("probs", {})
                    nli_conf = nli_probs.get(nli_label, nli_probs.get(nli_label.upper(), 0))

                    # LLM confidence
                    llm_probs = llm_results[i].get("probs", {})
                    llm_conf = llm_probs.get(llm_label.upper(), llm_probs.get(llm_label, 0))

                    # Case 1: confidence when assigning
                    conf_assigned["nli"][nli_label].append(nli_conf)
                    conf_assigned["llm"][llm_label].append(llm_conf)

                    # Case 2: confidence when agreeing
                    if nli_label == llm_label:
                        conf_agree[nli_label]["nli"].append(nli_conf)
                        conf_agree[nli_label]["llm"].append(llm_conf)

    # Save CSVs
    out_dir = os.path.join(results_dir, "CONFIDENCE_LLM-NLI", pipeline)
    os.makedirs(out_dir, exist_ok=True)

    # CSV 1: Confidence when assigning a label
    out1 = os.path.join(out_dir, "confidence_when_assigned.csv")
    with open(out1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "nli_count", "nli_avg_confidence", "llm_count", "llm_avg_confidence"])
        for label in label_names:
            nli_vals = conf_assigned["nli"].get(label, [])
            llm_vals = conf_assigned["llm"].get(label, [])
            w.writerow([
                label,
                len(nli_vals), round(np.mean(nli_vals), 4) if nli_vals else 0,
                len(llm_vals), round(np.mean(llm_vals), 4) if llm_vals else 0,
            ])
        all_nli = [v for vals in conf_assigned["nli"].values() for v in vals]
        all_llm = [v for vals in conf_assigned["llm"].values() for v in vals]
        w.writerow(["OVERALL", len(all_nli), round(np.mean(all_nli), 4), len(all_llm), round(np.mean(all_llm), 4)])

    # CSV 2: Confidence when both agree
    out2 = os.path.join(out_dir, "confidence_when_agree.csv")
    with open(out2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agreed_label", "count", "nli_avg_confidence", "llm_avg_confidence"])
        total_nli, total_llm, total_count = [], [], 0
        for label in label_names:
            nli_vals = conf_agree[label]["nli"]
            llm_vals = conf_agree[label]["llm"]
            count = len(nli_vals)
            w.writerow([
                label, count,
                round(np.mean(nli_vals), 4) if nli_vals else 0,
                round(np.mean(llm_vals), 4) if llm_vals else 0,
            ])
            total_nli.extend(nli_vals)
            total_llm.extend(llm_vals)
            total_count += count
        if total_count > 0:
            w.writerow(["OVERALL", total_count, round(np.mean(total_nli), 4), round(np.mean(total_llm), 4)])

    print(f"[{pipeline}] Saved: {out1}")
    print(f"[{pipeline}] Saved: {out2}")

print("\nDone!")
