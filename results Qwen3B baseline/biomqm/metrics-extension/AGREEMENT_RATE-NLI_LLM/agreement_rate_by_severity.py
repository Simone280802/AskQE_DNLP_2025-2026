"""
Agreement rate between NLI and LLM-Judge, broken down by severity.
For each severity: total question count, matches, and agreement rate.
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

# by_sev[severity] = {"total": int, "match": int}
by_sev = defaultdict(lambda: {"total": 0, "match": 0})

for lang in langs:
    nli_file = os.path.join(nli_dir, f"{lang}-nli.jsonl")
    llm_file = os.path.join(llm_dir, f"{lang}-llm-judge.jsonl")

    with open(nli_file) as fn, open(llm_file) as fl:
        for nli_line, llm_line in zip(fn, fl):
            nli_row = json.loads(nli_line)
            llm_row = json.loads(llm_line)

            nli_results = nli_row.get("nli_results", [])
            llm_results = llm_row.get("llm_judge_results", [])
            severities = nli_row.get("severities", [])

            n = min(len(nli_results), len(llm_results))
            for i in range(n):
                nli_label = nli_results[i]["label"].lower()
                llm_label = llm_results[i]["label"].lower()
                agree = (nli_label == llm_label)

                for sev in severities:
                    by_sev[sev]["total"] += 1
                    if agree:
                        by_sev[sev]["match"] += 1

# Save CSV
out_file = os.path.join(script_dir, "agreement_rate_by_severity.csv")
sev_order = ["Neutral", "Minor", "Major", "Critical"]

with open(out_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["severity", "matches", "total", "agreement_rate_pct"])
    overall_m, overall_t = 0, 0
    for sev in sev_order:
        d = by_sev.get(sev, {"total": 0, "match": 0})
        if d["total"] > 0:
            rate = round(d["match"] / d["total"] * 100, 2)
            w.writerow([sev, d["match"], d["total"], rate])
            overall_m += d["match"]
            overall_t += d["total"]
    w.writerow(["OVERALL", overall_m, overall_t, round(overall_m / overall_t * 100, 2)])

print(f"Saved: {out_file}")

# Print
print(f"\n{'Severity':<15} {'Matches':>10} {'Total':>10} {'Rate (%)':>10}")
print("-" * 47)
for sev in sev_order:
    d = by_sev.get(sev, {"total": 0, "match": 0})
    if d["total"] > 0:
        print(f"{sev:<15} {d['match']:>10} {d['total']:>10} {d['match']/d['total']*100:>10.2f}")
print("-" * 47)
print(f"{'OVERALL':<15} {overall_m:>10} {overall_t:>10} {overall_m/overall_t*100:>10.2f}")
