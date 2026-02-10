"""
Agreement Rate by Severity: computes how often NLI and LLM-Judge
produce the same label for each Q&A pair, broken down by severity.
"""

import json
import os
import csv

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

LLM_DIR = os.path.join(BASE_DIR, "llm-judge")
NLI_DIR = os.path.join(BASE_DIR, "nli")

# Stats: stats[severity] = {"matches": 0, "total": 0}
stats_global = {sev: {"matches": 0, "total": 0} for sev in SEVERITIES}
stats_by_lang = {
    lang: {sev: {"matches": 0, "total": 0} for sev in SEVERITIES}
    for lang in LANGUAGES
}

for lang in LANGUAGES:
    llm_path = os.path.join(LLM_DIR, f"{lang}-llm-judge.jsonl")
    nli_path = os.path.join(NLI_DIR, f"{lang}-nli.jsonl")

    if not os.path.exists(llm_path) or not os.path.exists(nli_path):
        print(f"SKIP {lang}: files not found")
        continue

    with open(llm_path, 'r') as f_llm, open(nli_path, 'r') as f_nli:
        for line_llm, line_nli in zip(f_llm, f_nli):
            row_llm = json.loads(line_llm)
            row_nli = json.loads(line_nli)

            severities = row_llm.get("severities", ["Neutral"])
            llm_results = row_llm.get("llm_judge_results", [])
            nli_results = row_nli.get("nli_results", [])

            # Compare each Q&A pair
            n_pairs = min(len(llm_results), len(nli_results))
            for i in range(n_pairs):
                llm_label = llm_results[i]["label"].upper()
                nli_label = nli_results[i]["label"].upper()
                match = 1 if llm_label == nli_label else 0

                for sev in severities:
                    if sev in SEVERITIES:
                        stats_global[sev]["matches"] += match
                        stats_global[sev]["total"] += 1
                        stats_by_lang[lang][sev]["matches"] += match
                        stats_by_lang[lang][sev]["total"] += 1

# ── Print results ──
print("=" * 60)
print("AGREEMENT RATE BY SEVERITY (GLOBAL)")
print("=" * 60)
print(f"{'Severity':<12} {'Matches':>8} {'Total':>8} {'Agreement':>10}")
print("-" * 42)
for sev in SEVERITIES:
    m = stats_global[sev]["matches"]
    t = stats_global[sev]["total"]
    rate = (m / t * 100) if t > 0 else 0
    print(f"{sev:<12} {m:>8} {t:>8} {rate:>9.2f}%")

total_m = sum(s["matches"] for s in stats_global.values())
total_t = sum(s["total"] for s in stats_global.values())
print(f"{'OVERALL':<12} {total_m:>8} {total_t:>8} {total_m/total_t*100:>9.2f}%")

print(f"\n{'=' * 60}")
print("AGREEMENT RATE BY LANGUAGE × SEVERITY")
print("=" * 60)
print(f"{'Language':<8} {'Severity':<12} {'Matches':>8} {'Total':>8} {'Agreement':>10}")
print("-" * 50)
for lang in LANGUAGES:
    for sev in SEVERITIES:
        m = stats_by_lang[lang][sev]["matches"]
        t = stats_by_lang[lang][sev]["total"]
        rate = (m / t * 100) if t > 0 else 0
        if t > 0:
            print(f"{lang:<8} {sev:<12} {m:>8} {t:>8} {rate:>9.2f}%")

# ── Save CSV ──
output_dir = os.path.dirname(BASE_DIR)
csv_path = os.path.join(output_dir, "agreement_by_severity.csv")

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Global
    writer.writerow(["GLOBAL AGREEMENT BY SEVERITY"])
    writer.writerow(["Severity", "Matches", "Total", "Agreement_Rate"])
    for sev in SEVERITIES:
        m = stats_global[sev]["matches"]
        t = stats_global[sev]["total"]
        rate = f"{m/t*100:.2f}%" if t > 0 else "N/A"
        writer.writerow([sev, m, t, rate])
    writer.writerow(["OVERALL", total_m, total_t, f"{total_m/total_t*100:.2f}%"])
    writer.writerow([])
    
    # By language × severity
    writer.writerow(["BY LANGUAGE × SEVERITY"])
    writer.writerow(["Language", "Severity", "Matches", "Total", "Agreement_Rate"])
    for lang in LANGUAGES:
        for sev in SEVERITIES:
            m = stats_by_lang[lang][sev]["matches"]
            t = stats_by_lang[lang][sev]["total"]
            rate = f"{m/t*100:.2f}%" if t > 0 else "N/A"
            if t > 0:
                writer.writerow([lang, sev, m, t, rate])

print(f"\nSaved: {csv_path}")
