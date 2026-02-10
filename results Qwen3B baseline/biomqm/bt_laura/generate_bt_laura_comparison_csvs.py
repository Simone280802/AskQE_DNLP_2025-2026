"""
Generate Comparison CSVs: Baseline vs bt_laura
Compares bt_laura against the Baseline for both string-comparison and SBERT metrics.
Outputs:
  - bt_laura_comparison_by_language.csv        (string comparison + SBERT, per language)
  - bt_laura_comparison_by_severity.csv        (string comparison + SBERT, per severity)
  - bt_laura_comparison_by_language_severity.csv (string comparison + SBERT, per lang-severity)
"""

import json
import os
import csv
from collections import defaultdict

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIOMQM_DIR = os.path.dirname(BASE_DIR)  # parent biomqm dir
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

# Paths
PATHS = {
    "baseline_string": os.path.join(BIOMQM_DIR, "baseline/evaluation/string comparison"),
    "baseline_sbert": os.path.join(BIOMQM_DIR, "baseline/evaluation/sbert"),
    "bt_laura_string": os.path.join(BASE_DIR, "QA/evaluation/string-comparison"),
    "bt_laura_sbert": os.path.join(BASE_DIR, "QA/evaluation/sbert"),
}


def load_string_data(source, lang):
    """Load string-comparison JSONL and aggregate metrics by severity."""
    if source == "baseline":
        filepath = os.path.join(PATHS["baseline_string"], f"{lang}-vanilla.jsonl")
    else:
        filepath = os.path.join(PATHS["bt_laura_string"], f"{lang}_lau_string.jsonl")

    if not os.path.exists(filepath):
        print(f"  String file not found: {filepath}")
        return {}

    stats = {sev: {"f1": [], "em": [], "chrf": [], "bleu": []} for sev in ALL_SEVERITIES}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                scores = row.get('scores', [])

                if not scores:
                    continue

                n = len(scores)
                avg_f1 = sum(s.get('f1', 0) for s in scores) / n
                avg_em = sum(1 if s.get('em', False) else 0 for s in scores) / n
                avg_chrf = sum(s.get('chrf', 0) for s in scores) / n
                avg_bleu = sum(s.get('bleu', 0) for s in scores) / n

                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        stats[sev]["f1"].append(avg_f1)
                        stats[sev]["em"].append(avg_em)
                        stats[sev]["chrf"].append(avg_chrf)
                        stats[sev]["bleu"].append(avg_bleu)
            except:
                continue

    return stats


def load_sbert_data(source, lang):
    """Load SBERT JSONL and aggregate similarity by severity."""
    if source == "baseline":
        filepath = os.path.join(PATHS["baseline_sbert"], f"{lang}-vanilla.jsonl")
    else:
        filepath = os.path.join(PATHS["bt_laura_sbert"], f"{lang}_lau.jsonl")

    if not os.path.exists(filepath):
        print(f"  SBERT file not found: {filepath}")
        return {}

    stats = {sev: {"sbert": []} for sev in ALL_SEVERITIES}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                scores = row.get('scores', [])

                if not scores:
                    continue

                n = len(scores)
                avg_sbert = sum(s.get('sbert_sim', 0) for s in scores) / n

                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        stats[sev]["sbert"].append(avg_sbert)
            except:
                continue

    return stats


def aggregate_stats(stats):
    """Calculate averages from lists of values."""
    result = {}
    for sev, metrics in stats.items():
        result[sev] = {}
        for metric_name, values in metrics.items():
            if values:
                result[sev][metric_name] = sum(values) / len(values)
                result[sev][f"{metric_name}_count"] = len(values)
            else:
                result[sev][metric_name] = None
                result[sev][f"{metric_name}_count"] = 0
    return result


def aggregate_overall(stats):
    """Aggregate all severities into overall metrics."""
    metric_names = set()
    for sev_metrics in stats.values():
        metric_names.update(k for k in sev_metrics.keys() if not k.endswith("_count"))

    combined = {m: [] for m in metric_names}
    for sev, metrics in stats.items():
        for m in metric_names:
            combined[m].extend(metrics.get(m, []))

    result = {}
    for metric_name, values in combined.items():
        if values:
            result[metric_name] = sum(values) / len(values)
            result[f"{metric_name}_count"] = len(values)
        else:
            result[metric_name] = None
            result[f"{metric_name}_count"] = 0
    return result


def format_value(v):
    if v is None:
        return "N/A"
    elif isinstance(v, float):
        return f"{v:.4f}"
    else:
        return v


def write_csv(rows, output_file):
    if not rows:
        print(f"  No rows to write for {output_file}")
        return
    fieldnames = list(rows[0].keys())
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(v) for k, v in row.items()})
    print(f"Saved: {output_file}")


# ── 1) Comparison by Language ───────────────────────────────────────
def generate_comparison_by_language():
    print("\n" + "="*60)
    print("Generating bt_laura_comparison_by_language.csv")
    print("="*60)

    rows = []
    for lang in LANGUAGES:
        print(f"  Processing {lang}...")
        row = {"Language": lang}

        # String comparison
        bl_str = aggregate_overall(load_string_data("baseline", lang))
        bt_str = aggregate_overall(load_string_data("bt_laura", lang))

        for m in ["f1", "em", "chrf", "bleu"]:
            mu = m.upper() if m in ("f1", "em") else ("chrF" if m == "chrf" else "BLEU")
            row[f"baseline_{mu}"] = bl_str.get(m)
            row[f"bt_laura_{mu}"] = bt_str.get(m)

        row["baseline_str_count"] = bl_str.get("f1_count", 0)
        row["bt_laura_str_count"] = bt_str.get("f1_count", 0)

        # SBERT
        bl_sb = aggregate_overall(load_sbert_data("baseline", lang))
        bt_sb = aggregate_overall(load_sbert_data("bt_laura", lang))

        row["baseline_SBERT"] = bl_sb.get("sbert")
        row["bt_laura_SBERT"] = bt_sb.get("sbert")
        row["baseline_sbert_count"] = bl_sb.get("sbert_count", 0)
        row["bt_laura_sbert_count"] = bt_sb.get("sbert_count", 0)

        # Deltas
        for m, mu in [("f1", "F1"), ("em", "EM"), ("chrf", "chrF"), ("bleu", "BLEU")]:
            bl_v = row.get(f"baseline_{mu}")
            bt_v = row.get(f"bt_laura_{mu}")
            row[f"delta_{mu}"] = (bt_v - bl_v) if (bt_v is not None and bl_v is not None) else None

        bl_sb_v = row.get("baseline_SBERT")
        bt_sb_v = row.get("bt_laura_SBERT")
        row["delta_SBERT"] = (bt_sb_v - bl_sb_v) if (bt_sb_v is not None and bl_sb_v is not None) else None

        rows.append(row)

    output = os.path.join(BASE_DIR, "bt_laura_comparison_by_language.csv")
    write_csv(rows, output)
    return rows


# ── 2) Comparison by Severity ──────────────────────────────────────
def generate_comparison_by_severity():
    print("\n" + "="*60)
    print("Generating bt_laura_comparison_by_severity.csv")
    print("="*60)

    # Accumulate across all languages per severity
    all_bl_str = {sev: {"f1": [], "em": [], "chrf": [], "bleu": []} for sev in ALL_SEVERITIES}
    all_bt_str = {sev: {"f1": [], "em": [], "chrf": [], "bleu": []} for sev in ALL_SEVERITIES}
    all_bl_sb = {sev: {"sbert": []} for sev in ALL_SEVERITIES}
    all_bt_sb = {sev: {"sbert": []} for sev in ALL_SEVERITIES}

    for lang in LANGUAGES:
        print(f"  Loading {lang}...")
        bl_str = load_string_data("baseline", lang)
        bt_str = load_string_data("bt_laura", lang)
        bl_sb = load_sbert_data("baseline", lang)
        bt_sb = load_sbert_data("bt_laura", lang)

        for sev in ALL_SEVERITIES:
            for m in ["f1", "em", "chrf", "bleu"]:
                all_bl_str[sev][m].extend(bl_str.get(sev, {}).get(m, []))
                all_bt_str[sev][m].extend(bt_str.get(sev, {}).get(m, []))
            all_bl_sb[sev]["sbert"].extend(bl_sb.get(sev, {}).get("sbert", []))
            all_bt_sb[sev]["sbert"].extend(bt_sb.get(sev, {}).get("sbert", []))

    rows = []
    for sev in ALL_SEVERITIES:
        row = {"Severity": sev}

        # String comparison averages
        for m, mu in [("f1", "F1"), ("em", "EM"), ("chrf", "chrF"), ("bleu", "BLEU")]:
            bl_vals = all_bl_str[sev][m]
            bt_vals = all_bt_str[sev][m]
            bl_avg = (sum(bl_vals) / len(bl_vals)) if bl_vals else None
            bt_avg = (sum(bt_vals) / len(bt_vals)) if bt_vals else None
            row[f"baseline_{mu}"] = bl_avg
            row[f"bt_laura_{mu}"] = bt_avg
            row[f"delta_{mu}"] = (bt_avg - bl_avg) if (bl_avg is not None and bt_avg is not None) else None

        row["baseline_str_count"] = len(all_bl_str[sev]["f1"])
        row["bt_laura_str_count"] = len(all_bt_str[sev]["f1"])

        # SBERT averages
        bl_sb_vals = all_bl_sb[sev]["sbert"]
        bt_sb_vals = all_bt_sb[sev]["sbert"]
        bl_sb_avg = (sum(bl_sb_vals) / len(bl_sb_vals)) if bl_sb_vals else None
        bt_sb_avg = (sum(bt_sb_vals) / len(bt_sb_vals)) if bt_sb_vals else None
        row["baseline_SBERT"] = bl_sb_avg
        row["bt_laura_SBERT"] = bt_sb_avg
        row["delta_SBERT"] = (bt_sb_avg - bl_sb_avg) if (bl_sb_avg is not None and bt_sb_avg is not None) else None
        row["baseline_sbert_count"] = len(bl_sb_vals)
        row["bt_laura_sbert_count"] = len(bt_sb_vals)

        rows.append(row)

    output = os.path.join(BASE_DIR, "bt_laura_comparison_by_severity.csv")
    write_csv(rows, output)
    return rows


# ── 3) Comparison by Language-Severity ─────────────────────────────
def generate_comparison_by_language_severity():
    print("\n" + "="*60)
    print("Generating bt_laura_comparison_by_language_severity.csv")
    print("="*60)

    rows = []
    for lang in LANGUAGES:
        print(f"  Processing {lang}...")

        bl_str = aggregate_stats(load_string_data("baseline", lang))
        bt_str = aggregate_stats(load_string_data("bt_laura", lang))
        bl_sb = aggregate_stats(load_sbert_data("baseline", lang))
        bt_sb = aggregate_stats(load_sbert_data("bt_laura", lang))

        for sev in ALL_SEVERITIES:
            row = {"Language": lang, "Severity": sev}

            # String comparison
            for m, mu in [("f1", "F1"), ("em", "EM"), ("chrf", "chrF"), ("bleu", "BLEU")]:
                bl_v = bl_str.get(sev, {}).get(m)
                bt_v = bt_str.get(sev, {}).get(m)
                row[f"baseline_{mu}"] = bl_v
                row[f"bt_laura_{mu}"] = bt_v
                row[f"delta_{mu}"] = (bt_v - bl_v) if (bl_v is not None and bt_v is not None) else None

            row["baseline_str_count"] = bl_str.get(sev, {}).get("f1_count", 0)
            row["bt_laura_str_count"] = bt_str.get(sev, {}).get("f1_count", 0)

            # SBERT
            bl_sb_v = bl_sb.get(sev, {}).get("sbert")
            bt_sb_v = bt_sb.get(sev, {}).get("sbert")
            row["baseline_SBERT"] = bl_sb_v
            row["bt_laura_SBERT"] = bt_sb_v
            row["delta_SBERT"] = (bt_sb_v - bl_sb_v) if (bl_sb_v is not None and bt_sb_v is not None) else None
            row["baseline_sbert_count"] = bl_sb.get(sev, {}).get("sbert_count", 0)
            row["bt_laura_sbert_count"] = bt_sb.get(sev, {}).get("sbert_count", 0)

            rows.append(row)

    output = os.path.join(BASE_DIR, "bt_laura_comparison_by_language_severity.csv")
    write_csv(rows, output)
    return rows


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Baseline vs bt_laura — Comparison CSV Generator")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Languages: {LANGUAGES}")
    print(f"Severities: {ALL_SEVERITIES}")

    generate_comparison_by_language()
    generate_comparison_by_severity()
    generate_comparison_by_language_severity()

    print("\n" + "="*60)
    print("Done! Generated 3 comparison CSVs in:")
    print(f"  {BASE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
