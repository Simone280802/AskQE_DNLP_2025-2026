"""
String Comparison: Baseline vs NER Extension
Adapted from evaluation/string-comparison/string_comparison.py

Reads pre-computed string comparison JSONL files.
- Baseline: scores list with f1/em/chrf/bleu per question
- NER extension: entity_metrics dict with f1/em/bleu/chrf per entity
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
metrics = ["f1", "em", "chrf", "bleu"]

# Paths to pre-computed string comparison evaluation results
BASELINE_DIR = os.path.join(biomqm_dir, "baseline", "evaluation", "string comparison")
NER_DIR = os.path.join(biomqm_dir, "ner-extension", "evaluation", "string-comparison")

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
    Read a string comparison JSONL file line by line.
    - Baseline: each score dict in scores list = 1 comparison
    - NER: each entity in entity_metrics = 1 comparison
    Returns list of (severity, {f1, em, chrf, bleu}) tuples.
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
                    # Baseline format: scores list with f1/em/chrf/bleu
                    scores = row.get("scores", [])
                    for score_entry in scores:
                        em_val = score_entry.get("em", False)
                        for sev in sevs:
                            results.append((sev, {
                                "f1": score_entry.get("f1", 0),
                                "em": 1 if em_val else 0,
                                "chrf": score_entry.get("chrf", 0),
                                "bleu": score_entry.get("bleu", 0),
                            }))
                else:
                    # NER format: entity_metrics dict with f1/em/bleu/chrf per entity
                    entity_metrics = row.get("entity_metrics", {})
                    for entity_name, entity_data in entity_metrics.items():
                        # Skip entities where BOTH answers are [NOT FOUND]
                        # (gives false 1.0). Keep one-sided [NOT FOUND] as
                        # they represent real translation failures.
                        a_src = str(entity_data.get("answer_src", ""))
                        a_bt = str(entity_data.get("answer_bt", ""))
                        if a_src == "[NOT FOUND]" and a_bt == "[NOT FOUND]":
                            continue
                        em_val = entity_data.get("em", 0)
                        for sev in sevs:
                            results.append((sev, {
                                "f1": entity_data.get("f1", 0),
                                "em": 1 if em_val else 0,
                                "chrf": entity_data.get("chrf", 0),
                                "bleu": entity_data.get("bleu", 0),
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

        print(f"{strat:<16} {lang:<8} entries={len(entries)}")


# ── 1. Output: string_comparison_by_severity.csv ──
out_file = os.path.join(OUTPUT_DIR, "string_comparison_by_severity.csv")

header = ["severity", "count_baseline", "count_ner"]
for m in metrics:
    for s in strategies:
        header.append(f"{s}_{m}")
    header.append(f"delta_ner_{m}_pct")

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

        count_bl = count_by_severity["baseline"][sev]
        count_ner = count_by_severity["ner-extension"][sev]

        row = [sev, count_bl, count_ner]
        for m in metrics:
            bl_avg = np.mean(by_severity["baseline"][sev][m]) if by_severity["baseline"][sev][m] else 0
            ner_avg = np.mean(by_severity["ner-extension"][sev][m]) if by_severity["ner-extension"][sev][m] else 0
            delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0
            row.extend([round(bl_avg, 4), round(ner_avg, 4), delta])

        writer.writerow(row)

        for s in strategies:
            for m in metrics:
                overall[s][m].extend(by_severity[s][sev][m])
            overall_count[s] += count_by_severity[s][sev]

        bl = {m: np.mean(by_severity["baseline"][sev][m]) for m in metrics}
        print(f"{sev:<10} bl={count_bl:<6} ner={count_ner:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

    # OVERALL
    row = ["OVERALL", overall_count["baseline"], overall_count["ner-extension"]]
    for m in metrics:
        bl_avg = np.mean(overall["baseline"][m]) if overall["baseline"][m] else 0
        ner_avg = np.mean(overall["ner-extension"][m]) if overall["ner-extension"][m] else 0
        delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0
        row.extend([round(bl_avg, 4), round(ner_avg, 4), delta])
    writer.writerow(row)

    bl = {m: np.mean(overall["baseline"][m]) for m in metrics}
    print(f"{'OVERALL':<10} bl={overall_count['baseline']:<6} ner={overall_count['ner-extension']:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

print(f"\nSaved: {out_file}")


# ── 2. Output: string_comparison_by_language_severity.csv ──
out_file2 = os.path.join(OUTPUT_DIR, "string_comparison_by_language_severity.csv")

header2 = ["language", "severity", "count_baseline", "count_ner"]
for m in metrics:
    for s in strategies:
        header2.append(f"{s}_{m}")
    header2.append(f"delta_ner_{m}_pct")

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

            count_bl = count_by_lang_severity["baseline"][(lang, sev)]
            count_ner = count_by_lang_severity["ner-extension"][(lang, sev)]

            row = [lang, sev, count_bl, count_ner]
            for m in metrics:
                bl_avg = np.mean(by_lang_severity["baseline"][(lang, sev)][m]) if by_lang_severity["baseline"][(lang, sev)][m] else 0
                ner_avg = np.mean(by_lang_severity["ner-extension"][(lang, sev)][m]) if by_lang_severity["ner-extension"][(lang, sev)][m] else 0
                delta = round(((ner_avg - bl_avg) / bl_avg) * 100, 2) if bl_avg != 0 else 0
                row.extend([round(bl_avg, 4), round(ner_avg, 4), delta])
            writer.writerow(row)

            bl = {m: np.mean(by_lang_severity["baseline"][(lang, sev)][m]) for m in metrics}
            print(f"{lang:<8} {sev:<10} bl={count_bl:<6} ner={count_ner:<6} F1={bl['f1']:.4f}  EM={bl['em']:.4f}  chrF={bl['chrf']:.2f}  BLEU={bl['bleu']:.2f}")

print(f"\nSaved: {out_file2}")
