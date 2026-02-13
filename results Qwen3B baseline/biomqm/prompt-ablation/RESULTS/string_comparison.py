"""
String Comparison: Baseline vs Prompt Ablation Strategies (P1, P2, P3)
Adapted from evaluation/string-comparison/string_comparison.py

Reads pre-computed string comparison JSONL files, treats each individual
score (f1, em, chrf, bleu) as a separate comparison (like the original
string_comparison.py), groups by severity, and outputs comparison CSVs
with percentage deltas.
"""

import json
import csv
import os
import numpy as np
from collections import defaultdict

# ── Configuration ──
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = .../prompt-ablation/RESULTS
# Go up 2 levels: RESULTS -> prompt-ablation -> biomqm
biomqm_dir = os.path.dirname(os.path.dirname(script_dir))

languages = ["de", "es", "fr", "ru", "zh-CN"]
severities_order = ["Neutral", "Minor", "Major", "Critical"]
metrics = ["f1", "em", "chrf", "bleu"]

# Paths to pre-computed string comparison evaluation results
# Note: baseline uses "string comparison" (space), PA uses "string-comparison" (dash)
BASELINE_DIR = os.path.join(biomqm_dir, "baseline", "evaluation", "string comparison")
PA_DIRS = {
    "P1": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P1-fewshot", "mapped", "evaluation", "string-comparison"),
    "P2": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P2-cot", "mapped", "evaluation", "string-comparison"),
    "P3": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P3-concise", "mapped", "evaluation", "string-comparison"),
}

strategies = ["baseline", "P1", "P2", "P3"]

OUTPUT_DIR = script_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_filepath(strategy, lang):
    """Return the JSONL file path for a given strategy and language."""
    if strategy == "baseline":
        return os.path.join(BASELINE_DIR, f"{lang}-vanilla.jsonl")
    else:
        return os.path.join(PA_DIRS[strategy], f"{lang}.jsonl")


def collect_scores(strategy, lang):
    """
    Read a string comparison JSONL file line by line (like original string_comparison.py).
    Each individual score dict (f1, em, chrf, bleu) is a separate comparison.
    Returns a list of (severity, {f1, em, chrf, bleu}) tuples.
    """
    filepath = get_filepath(strategy, lang)
    results = []

    if not os.path.exists(filepath):
        print(f"WARNING: File not found: {filepath}")
        return results

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                sevs = row.get("severities", [])
                scores = row.get("scores", [])

                # Each individual score is a separate comparison
                # (like string_comparison.py: for pred, ref in zip(...))
                for score_entry in scores:
                    f1 = score_entry.get("f1", 0)
                    em = score_entry.get("em", False)
                    chrf = score_entry.get("chrf", 0)
                    bleu = score_entry.get("bleu", 0)

                    for sev in sevs:
                        results.append((sev, {
                            "f1": f1,
                            "em": 1 if em else 0,  # convert bool to int
                            "chrf": chrf,
                            "bleu": bleu
                        }))

            except json.JSONDecodeError as e:
                print(f"Skipping corrupted line: {e}")
                continue

    return results


# ── Collect all scores ──
by_severity = {s: defaultdict(lambda: defaultdict(list)) for s in strategies}
by_lang_severity = {s: defaultdict(lambda: defaultdict(list)) for s in strategies}
count_by_severity = {s: defaultdict(int) for s in strategies}
count_by_lang_severity = {s: defaultdict(int) for s in strategies}

for lang in languages:
    for strat in strategies:
        entries = collect_scores(strat, lang)
        for sev, score_dict in entries:
            for m in metrics:
                by_severity[strat][sev][m].append(score_dict[m])
                by_lang_severity[strat][(lang, sev)][m].append(score_dict[m])
            count_by_severity[strat][sev] += 1
            count_by_lang_severity[strat][(lang, sev)] += 1

        print(f"{strat:<10} {lang:<8} entries={len(entries)}")


# ── 1. Output: string_comparison_by_severity.csv ──
out_file = os.path.join(OUTPUT_DIR, "string_comparison_by_severity.csv")

header = ["severity", "count"]
for m in metrics:
    for s in strategies:
        header.append(f"{s}_{m}")
    for p in ["P1", "P2", "P3"]:
        header.append(f"delta_{p}_{m}_pct")

print("\n" + "=" * 80)
print("STRING COMPARISON BY SEVERITY")
print("=" * 80)

with open(out_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    overall = {s: defaultdict(list) for s in strategies}
    overall_count = {s: 0 for s in strategies}

    for sev in severities_order:
        if not by_severity["baseline"][sev]["f1"]:
            continue

        count = count_by_severity["baseline"][sev]

        row = [sev, count]
        for m in metrics:
            avgs = {}
            for s in strategies:
                vals = by_severity[s][sev][m]
                avgs[s] = np.mean(vals) if vals else 0
                row.append(round(avgs[s], 4))
            for p in ["P1", "P2", "P3"]:
                delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
                row.append(delta)

        writer.writerow(row)

        for s in strategies:
            for m in metrics:
                overall[s][m].extend(by_severity[s][sev][m])
            overall_count[s] += count_by_severity[s][sev]

        # Print summary
        bl = {m: np.mean(by_severity["baseline"][sev][m]) for m in metrics}
        print(f"{sev:<10} count={count:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

    # OVERALL row
    count = overall_count["baseline"]
    row = ["OVERALL", count]
    for m in metrics:
        avgs = {}
        for s in strategies:
            vals = overall[s][m]
            avgs[s] = np.mean(vals) if vals else 0
            row.append(round(avgs[s], 4))
        for p in ["P1", "P2", "P3"]:
            delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
            row.append(delta)
    writer.writerow(row)

    bl = {m: np.mean(overall["baseline"][m]) for m in metrics}
    print(f"{'OVERALL':<10} count={count:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

print(f"\nSaved: {out_file}")


# ── 2. Output: string_comparison_by_language_severity.csv ──
out_file2 = os.path.join(OUTPUT_DIR, "string_comparison_by_language_severity.csv")

header2 = ["language", "severity", "count"]
for m in metrics:
    for s in strategies:
        header2.append(f"{s}_{m}")
    for p in ["P1", "P2", "P3"]:
        header2.append(f"delta_{p}_{m}_pct")

print("\n" + "=" * 80)
print("STRING COMPARISON BY LANGUAGE-SEVERITY")
print("=" * 80)

with open(out_file2, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header2)

    for lang in languages:
        for sev in severities_order:
            if not by_lang_severity["baseline"][(lang, sev)]["f1"]:
                continue

            count = count_by_lang_severity["baseline"][(lang, sev)]

            row = [lang, sev, count]
            for m in metrics:
                avgs = {}
                for s in strategies:
                    vals = by_lang_severity[s][(lang, sev)][m]
                    avgs[s] = np.mean(vals) if vals else 0
                    row.append(round(avgs[s], 4))
                for p in ["P1", "P2", "P3"]:
                    delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
                    row.append(delta)
            writer.writerow(row)

            bl = {m: np.mean(by_lang_severity["baseline"][(lang, sev)][m]) for m in metrics}
            print(f"{lang:<8} {sev:<10} count={count:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

print(f"\nSaved: {out_file2}")
