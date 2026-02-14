"""
SBERT Comparison: Baseline vs Prompt Ablation Strategies (P1, P2, P3)
Adapted from evaluation/sbert/sbert.py

Reads pre-computed SBERT evaluation JSONL files, treats each individual
sbert_sim score as a separate comparison (like the original sbert.py),
groups by severity, and outputs comparison CSVs with percentage deltas.
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

# Paths to pre-computed SBERT evaluation results
BASELINE_DIR = os.path.join(biomqm_dir, "baseline", "evaluation", "sbert")
PA_DIRS = {
    "P1": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P1-fewshot", "mapped", "evaluation", "sbert"),
    "P2": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P2-cot", "mapped", "evaluation", "sbert"),
    "P3": os.path.join(biomqm_dir, "prompt-ablation", "QA", "P3-concise", "mapped", "evaluation", "sbert"),
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
    Read a SBERT JSONL file line by line (like original sbert.py).
    Each individual sbert_sim score is a separate comparison.
    Returns a list of (severity, score) tuples.
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
                for score_entry in scores:
                    sbert_sim = score_entry.get("sbert_sim", None)
                    if sbert_sim is not None:
                        for sev in sevs:
                            results.append((sev, sbert_sim))

            except json.JSONDecodeError as e:
                print(f"Skipping corrupted line: {e}")
                continue

    return results


# ── Collect all scores ──
# by_severity[strategy][severity] = [list of individual sbert_sim scores]
by_severity = {s: defaultdict(list) for s in strategies}
# by_lang_severity[strategy][(lang, severity)] = [list of individual sbert_sim scores]
by_lang_severity = {s: defaultdict(list) for s in strategies}

for lang in languages:
    for strat in strategies:
        entries = collect_scores(strat, lang)
        for sev, score in entries:
            by_severity[strat][sev].append(score)
            by_lang_severity[strat][(lang, sev)].append(score)

        print(f"{strat:<10} {lang:<8} entries={len(entries)}")


# ── 1. Output: sbert_by_severity.csv ──
out_file = os.path.join(OUTPUT_DIR, "sbert_by_severity.csv")

header = ["severity", "count"]
for s in strategies:
    header.append(f"{s}_sbert")
for p in ["P1", "P2", "P3"]:
    header.append(f"delta_{p}_pct")

print("\n" + "=" * 80)
print("SBERT BY SEVERITY")
print("=" * 80)

with open(out_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    overall = {s: [] for s in strategies}

    for sev in severities_order:
        baseline_scores = by_severity["baseline"].get(sev, [])
        if not baseline_scores:
            continue

        count = len(baseline_scores)
        avgs = {}
        for s in strategies:
            scores = by_severity[s].get(sev, [])
            avgs[s] = np.mean(scores) if scores else 0

        row = [sev, count]
        for s in strategies:
            row.append(round(avgs[s], 4))
        for p in ["P1", "P2", "P3"]:
            delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
            row.append(delta)
        writer.writerow(row)

        for s in strategies:
            overall[s].extend(by_severity[s].get(sev, []))

        print(f"{sev:<10} count={count:<6} baseline={avgs['baseline']:.4f}  P1={avgs['P1']:.4f}  P2={avgs['P2']:.4f}  P3={avgs['P3']:.4f}")

    # OVERALL row
    avgs = {s: np.mean(overall[s]) if overall[s] else 0 for s in strategies}
    row = ["OVERALL", len(overall["baseline"])]
    for s in strategies:
        row.append(round(avgs[s], 4))
    for p in ["P1", "P2", "P3"]:
        delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
        row.append(delta)
    writer.writerow(row)
    print(f"{'OVERALL':<10} count={len(overall['baseline']):<6} baseline={avgs['baseline']:.4f}  P1={avgs['P1']:.4f}  P2={avgs['P2']:.4f}  P3={avgs['P3']:.4f}")

print(f"\nSaved: {out_file}")


# ── 2. Output: sbert_by_language_severity.csv ──
out_file2 = os.path.join(OUTPUT_DIR, "sbert_by_language_severity.csv")

header2 = ["language", "severity", "count"]
for s in strategies:
    header2.append(f"{s}_sbert")
for p in ["P1", "P2", "P3"]:
    header2.append(f"delta_{p}_pct")

print("\n" + "=" * 80)
print("SBERT BY LANGUAGE-SEVERITY")
print("=" * 80)

with open(out_file2, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header2)

    for lang in languages:
        for sev in severities_order:
            baseline_scores = by_lang_severity["baseline"].get((lang, sev), [])
            if not baseline_scores:
                continue

            count = len(baseline_scores)
            avgs = {}
            for s in strategies:
                scores = by_lang_severity[s].get((lang, sev), [])
                avgs[s] = np.mean(scores) if scores else 0

            row = [lang, sev, count]
            for s in strategies:
                row.append(round(avgs[s], 4))
            for p in ["P1", "P2", "P3"]:
                delta = round(((avgs[p] - avgs["baseline"]) / avgs["baseline"]) * 100, 2) if avgs["baseline"] != 0 else 0
                row.append(delta)
            writer.writerow(row)

            print(f"{lang:<8} {sev:<10} count={count:<6} baseline={avgs['baseline']:.4f}  P1={avgs['P1']:.4f}  P2={avgs['P2']:.4f}  P3={avgs['P3']:.4f}")

print(f"\nSaved: {out_file2}")
