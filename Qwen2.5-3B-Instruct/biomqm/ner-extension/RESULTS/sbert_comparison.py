"""
SBERT Comparison: Baseline vs NER Extension
Adapted from evaluation/sbert/sbert.py

Reads pre-computed SBERT evaluation JSONL files.
- Baseline: scores list with sbert_sim per question
- NER extension: entity_metrics dict with similarity per entity
Each individual score is a separate comparison.
Groups by severity and outputs comparison CSVs with percentage deltas.
"""

import json
import csv
import os
import numpy as np
from collections import defaultdict

# ── Configuration ──
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = .../ner-extension/RESULTS
# Go up 2 levels: RESULTS -> ner-extension -> biomqm
biomqm_dir = os.path.dirname(os.path.dirname(script_dir))

languages = ["de", "es", "fr", "ru", "zh-CN"]
severities_order = ["Neutral", "Minor", "Major", "Critical"]

# Paths to pre-computed SBERT evaluation results
BASELINE_DIR = os.path.join(biomqm_dir, "baseline", "evaluation", "sbert")
NER_DIR = os.path.join(biomqm_dir, "ner-extension", "evaluation", "sbert")

strategies = ["baseline", "ner-extension"]

OUTPUT_DIR = script_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_filepath(strategy, lang):
    if strategy == "baseline":
        return os.path.join(BASELINE_DIR, f"{lang}-vanilla.jsonl")
    else:
        return os.path.join(NER_DIR, f"{lang}.jsonl")


def collect_scores(strategy, lang):
    """
    Read a SBERT evaluation JSONL file line by line.
    - Baseline: each sbert_sim in scores list = 1 comparison
    - NER: each entity similarity in entity_metrics = 1 comparison
    Returns list of (severity, score) tuples.
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

                if strategy == "baseline":
                    # Baseline format: scores list with sbert_sim
                    scores = row.get("scores", [])
                    for score_entry in scores:
                        sbert_sim = score_entry.get("sbert_sim", None)
                        if sbert_sim is not None:
                            for sev in sevs:
                                results.append((sev, sbert_sim))
                else:
                    # NER format: entity_metrics dict with similarity per entity
                    entity_metrics = row.get("entity_metrics", {})
                    for entity_name, entity_data in entity_metrics.items():
                        # Skip entities where BOTH answers are [NOT FOUND]
                        # (gives false 1.0). Keep one-sided [NOT FOUND] as
                        # they represent real translation failures.
                        a_src = str(entity_data.get("answer_src", ""))
                        a_bt = str(entity_data.get("answer_bt", ""))
                        if a_src == "[NOT FOUND]" and a_bt == "[NOT FOUND]":
                            continue
                        sim = entity_data.get("similarity", None)
                        if sim is not None:
                            for sev in sevs:
                                results.append((sev, sim))

            except json.JSONDecodeError as e:
                print(f"Skipping corrupted line: {e}")
                continue

    return results


# ── Collect all scores ──
by_severity = {s: defaultdict(list) for s in strategies}
by_lang_severity = {s: defaultdict(list) for s in strategies}

for lang in languages:
    for strat in strategies:
        entries = collect_scores(strat, lang)
        for sev, score in entries:
            by_severity[strat][sev].append(score)
            by_lang_severity[strat][(lang, sev)].append(score)
        print(f"{strat:<16} {lang:<8} entries={len(entries)}")


# ── 1. Output: sbert_by_severity.csv ──
out_file = os.path.join(OUTPUT_DIR, "sbert_by_severity.csv")

header = ["severity", "count_baseline", "count_ner"]
for s in strategies:
    header.append(f"{s}_sbert")
header.append("delta_ner_pct")

print("\n" + "=" * 80)
print("SBERT BY SEVERITY")
print("=" * 80)

with open(out_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    overall = {s: [] for s in strategies}

    for sev in severities_order:
        bl_scores = by_severity["baseline"].get(sev, [])
        ner_scores = by_severity["ner-extension"].get(sev, [])
        if not bl_scores:
            continue

        bl_avg = np.mean(bl_scores) if bl_scores else 0
        ner_avg = np.mean(ner_scores) if ner_scores else 0
        delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0

        row = [sev, len(bl_scores), len(ner_scores), round(bl_avg, 4), round(ner_avg, 4), delta]
        writer.writerow(row)

        overall["baseline"].extend(bl_scores)
        overall["ner-extension"].extend(ner_scores)

        print(f"{sev:<10} bl_count={len(bl_scores):<6} ner_count={len(ner_scores):<6} baseline={bl_avg:.4f}  ner={ner_avg:.4f}  delta={delta:+.2f}%")

    # OVERALL
    bl_avg = np.mean(overall["baseline"]) if overall["baseline"] else 0
    ner_avg = np.mean(overall["ner-extension"]) if overall["ner-extension"] else 0
    delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0
    row = ["OVERALL", len(overall["baseline"]), len(overall["ner-extension"]), round(bl_avg, 4), round(ner_avg, 4), delta]
    writer.writerow(row)
    print(f"{'OVERALL':<10} bl_count={len(overall['baseline']):<6} ner_count={len(overall['ner-extension']):<6} baseline={bl_avg:.4f}  ner={ner_avg:.4f}  delta={delta:+.2f}%")

print(f"\nSaved: {out_file}")


# ── 2. Output: sbert_by_language_severity.csv ──
out_file2 = os.path.join(OUTPUT_DIR, "sbert_by_language_severity.csv")

header2 = ["language", "severity", "count_baseline", "count_ner"]
for s in strategies:
    header2.append(f"{s}_sbert")
header2.append("delta_ner_pct")

print("\n" + "=" * 80)
print("SBERT BY LANGUAGE-SEVERITY")
print("=" * 80)

with open(out_file2, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header2)

    for lang in languages:
        for sev in severities_order:
            bl_scores = by_lang_severity["baseline"].get((lang, sev), [])
            ner_scores = by_lang_severity["ner-extension"].get((lang, sev), [])
            if not bl_scores:
                continue

            bl_avg = np.mean(bl_scores) if bl_scores else 0
            ner_avg = np.mean(ner_scores) if ner_scores else 0
            delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0

            row = [lang, sev, len(bl_scores), len(ner_scores), round(bl_avg, 4), round(ner_avg, 4), delta]
            writer.writerow(row)

            print(f"{lang:<8} {sev:<10} bl={len(bl_scores):<6} ner={len(ner_scores):<6} baseline={bl_avg:.4f}  ner={ner_avg:.4f}  delta={delta:+.2f}%")

print(f"\nSaved: {out_file2}")
