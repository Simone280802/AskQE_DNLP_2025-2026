"""
Agreement Rate between NLI and LLM-Judge for ContraTICO dataset.
Broken down by perturbation type, computed per pipeline (atomic, semantic, vanilla).
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

for pipeline in pipelines:
    nli_dir = os.path.join(nli_base, pipeline)
    llm_dir = os.path.join(llm_base, pipeline)

    # by_pert[perturbation] = {"total": int, "match": int}
    by_pert = defaultdict(lambda: {"total": 0, "match": 0})

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
                perturbation = nli_row.get("perturbation", "unknown")

                n = min(len(nli_results), len(llm_results))
                for i in range(n):
                    nli_label = nli_results[i]["label"].lower()
                    llm_label = llm_results[i]["label"].lower()
                    agree = (nli_label == llm_label)

                    by_pert[perturbation]["total"] += 1
                    if agree:
                        by_pert[perturbation]["match"] += 1

    # Save CSV
    out_dir = os.path.join(results_dir, "AGREEMENT_RATE_LLM-NLI", pipeline)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "agreement_rate_by_perturbation.csv")

    pert_order = sorted(by_pert.keys())

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "matches", "total", "agreement_rate_pct"])
        overall_m, overall_t = 0, 0
        for pert in pert_order:
            d = by_pert[pert]
            if d["total"] > 0:
                rate = round(d["match"] / d["total"] * 100, 2)
                w.writerow([pert, d["match"], d["total"], rate])
                overall_m += d["match"]
                overall_t += d["total"]
        if overall_t > 0:
            w.writerow(["OVERALL", overall_m, overall_t, round(overall_m / overall_t * 100, 2)])

    print(f"[{pipeline}] Saved: {out_file}")
    print(f"  Perturbations: {len(pert_order)}, Total pairs: {overall_t}, Agreement: {round(overall_m/overall_t*100, 2) if overall_t else 0}%")

print("\nDone!")
