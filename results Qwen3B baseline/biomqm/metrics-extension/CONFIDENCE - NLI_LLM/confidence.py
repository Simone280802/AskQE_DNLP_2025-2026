"""
Confidence analysis for NLI and LLM-Judge models.

1. Average confidence when assigning each label (how confident each model is
   in its own prediction, regardless of whether the other model agrees).
2. Average confidence when both models agree on the same label.

Confidence = probability assigned to the chosen label.
"""
import json
import os
import csv
import numpy as np
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(os.path.dirname(script_dir), "results")
nli_dir = os.path.join(base, "nli")
llm_dir = os.path.join(base, "llm-judge")

langs = ["de", "es", "fr", "ru", "zh-CN"]
labels = ["entailment", "neutral", "contradiction"]

# ── 1. Confidence when assigning a label ──
# conf_assigned[model][label] = list of confidence values
conf_assigned = {
    "nli": defaultdict(list),
    "llm": defaultdict(list),
}

# ── 2. Confidence when both agree ──
# conf_agree[label] = {"nli": [...], "llm": [...]}
conf_agree = {label: {"nli": [], "llm": []} for label in labels}

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

                # NLI confidence: prob of its chosen label
                nli_probs = nli_results[i].get("probs", {})
                nli_conf = nli_probs.get(nli_label, nli_probs.get(nli_label.upper(), 0))

                # LLM confidence: prob of its chosen label
                llm_probs = llm_results[i].get("probs", {})
                llm_conf = llm_probs.get(llm_label.upper(), llm_probs.get(llm_label, 0))

                # Case 1: confidence when assigning
                conf_assigned["nli"][nli_label].append(nli_conf)
                conf_assigned["llm"][llm_label].append(llm_conf)

                # Case 2: confidence when agreeing
                if nli_label == llm_label:
                    conf_agree[nli_label]["nli"].append(nli_conf)
                    conf_agree[nli_label]["llm"].append(llm_conf)

# ── Save CSV 1: Confidence when assigning a label ──
out1 = os.path.join(script_dir, "confidence_when_assigned.csv")
with open(out1, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "nli_count", "nli_avg_confidence", "llm_count", "llm_avg_confidence"])
    for label in labels:
        nli_vals = conf_assigned["nli"].get(label, [])
        llm_vals = conf_assigned["llm"].get(label, [])
        w.writerow([
            label,
            len(nli_vals), round(np.mean(nli_vals), 4) if nli_vals else 0,
            len(llm_vals), round(np.mean(llm_vals), 4) if llm_vals else 0,
        ])
    # Overall
    all_nli = [v for vals in conf_assigned["nli"].values() for v in vals]
    all_llm = [v for vals in conf_assigned["llm"].values() for v in vals]
    w.writerow(["OVERALL", len(all_nli), round(np.mean(all_nli), 4), len(all_llm), round(np.mean(all_llm), 4)])

# ── Save CSV 2: Confidence when both agree ──
out2 = os.path.join(script_dir, "confidence_when_agree.csv")
with open(out2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["agreed_label", "count", "nli_avg_confidence", "llm_avg_confidence"])
    total_nli, total_llm, total_count = [], [], 0
    for label in labels:
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
    w.writerow(["OVERALL", total_count, round(np.mean(total_nli), 4), round(np.mean(total_llm), 4)])

print(f"Saved: {out1}")
print(f"Saved: {out2}")

# ── Print results ──
print("\n" + "=" * 60)
print("CONFIDENCE WHEN ASSIGNING A LABEL")
print("=" * 60)
print(f"{'Label':<18} {'NLI count':>10} {'NLI conf':>10} {'LLM count':>10} {'LLM conf':>10}")
print("-" * 60)
for label in labels:
    nv = conf_assigned["nli"].get(label, [])
    lv = conf_assigned["llm"].get(label, [])
    print(f"{label:<18} {len(nv):>10} {np.mean(nv):>10.4f} {len(lv):>10} {np.mean(lv):>10.4f}")
print(f"{'OVERALL':<18} {len(all_nli):>10} {np.mean(all_nli):>10.4f} {len(all_llm):>10} {np.mean(all_llm):>10.4f}")

print("\n" + "=" * 60)
print("CONFIDENCE WHEN BOTH AGREE ON THE SAME LABEL")
print("=" * 60)
print(f"{'Agreed Label':<18} {'Count':>8} {'NLI conf':>10} {'LLM conf':>10}")
print("-" * 48)
for label in labels:
    nv = conf_agree[label]["nli"]
    lv = conf_agree[label]["llm"]
    print(f"{label:<18} {len(nv):>8} {np.mean(nv):>10.4f} {np.mean(lv):>10.4f}")
print(f"{'OVERALL':<18} {total_count:>8} {np.mean(total_nli):>10.4f} {np.mean(total_llm):>10.4f}")
