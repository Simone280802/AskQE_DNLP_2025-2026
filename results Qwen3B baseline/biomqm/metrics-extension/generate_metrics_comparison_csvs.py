"""
Generate Metrics Extension Comparison CSVs: LLM-Judge vs NLI
Compares LLM-Judge and NLI model predictions.
Outputs three CSVs:
  1. comparison_by_language.csv - Aggregated by language
  2. comparison_by_language_severity.csv - Aggregated by language and severity
  3. label_agreement.csv - Global label match counts between LLM and NLI
"""

import json
import os
import csv
from collections import defaultdict

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]
LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


def load_llm_judge_data(lang):
    """Load LLM-Judge JSONL file."""
    filepath = os.path.join(RESULTS_DIR, "llm-judge", f"{lang}-llm-judge.jsonl")
    if not os.path.exists(filepath):
        print(f"  LLM-Judge file not found: {filepath}")
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                data.append(row)
            except json.JSONDecodeError:
                continue
    return data


def load_nli_data(lang):
    """Load NLI JSONL file."""
    filepath = os.path.join(RESULTS_DIR, "nli", f"{lang}-nli.jsonl")
    if not os.path.exists(filepath):
        print(f"  NLI file not found: {filepath}")
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                data.append(row)
            except json.JSONDecodeError:
                continue
    return data


def normalize_label(label):
    """Normalize label to uppercase."""
    return label.upper() if label else None


def get_probs_from_result(result, model_type):
    """Extract probabilities from a result dict, normalizing keys."""
    probs = result.get("probs", {})
    if model_type == "llm":
        # LLM uses uppercase keys
        return {
            "ENTAILMENT": probs.get("ENTAILMENT", 0),
            "NEUTRAL": probs.get("NEUTRAL", 0),
            "CONTRADICTION": probs.get("CONTRADICTION", 0)
        }
    else:
        # NLI uses lowercase keys
        return {
            "ENTAILMENT": probs.get("entailment", 0),
            "NEUTRAL": probs.get("neutral", 0),
            "CONTRADICTION": probs.get("contradiction", 0)
        }


def aggregate_row_metrics(results, model_type):
    """Aggregate metrics for a single row (sentence)."""
    if not results:
        return None
    
    label_counts = defaultdict(int)
    prob_sums = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
    n = len(results)
    
    for result in results:
        label = normalize_label(result.get("label", ""))
        if label in LABELS:
            label_counts[label] += 1
        
        probs = get_probs_from_result(result, model_type)
        for lbl in LABELS:
            prob_sums[lbl] += probs[lbl]
    
    # Calculate averages and percentages
    return {
        "count": n,
        "entailment_pct": label_counts["ENTAILMENT"] / n if n > 0 else 0,
        "neutral_pct": label_counts["NEUTRAL"] / n if n > 0 else 0,
        "contradiction_pct": label_counts["CONTRADICTION"] / n if n > 0 else 0,
        "avg_entailment_prob": prob_sums["ENTAILMENT"] / n if n > 0 else 0,
        "avg_neutral_prob": prob_sums["NEUTRAL"] / n if n > 0 else 0,
        "avg_contradiction_prob": prob_sums["CONTRADICTION"] / n if n > 0 else 0,
        "label_counts": dict(label_counts)
    }


def generate_comparison_by_language():
    """Generate CSV comparing LLM vs NLI metrics per language."""
    print("\n" + "="*60)
    print("Generating metrics_comparison_by_language.csv")
    print("="*60)
    
    rows = []
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        
        llm_data = load_llm_judge_data(lang)
        nli_data = load_nli_data(lang)
        
        if not llm_data or not nli_data:
            print(f"  Skipping {lang} - missing data")
            continue
        
        # Aggregate all rows for LLM
        llm_metrics = {
            "entailment_pct": [], "neutral_pct": [], "contradiction_pct": [],
            "avg_entailment_prob": [], "avg_neutral_prob": [], "avg_contradiction_prob": []
        }
        
        # Aggregate all rows for NLI
        nli_metrics = {
            "entailment_pct": [], "neutral_pct": [], "contradiction_pct": [],
            "avg_entailment_prob": [], "avg_neutral_prob": [], "avg_contradiction_prob": []
        }
        
        for llm_row in llm_data:
            results = llm_row.get("llm_judge_results", [])
            agg = aggregate_row_metrics(results, "llm")
            if agg:
                for key in llm_metrics:
                    llm_metrics[key].append(agg[key])
        
        for nli_row in nli_data:
            results = nli_row.get("nli_results", [])
            agg = aggregate_row_metrics(results, "nli")
            if agg:
                for key in nli_metrics:
                    nli_metrics[key].append(agg[key])
        
        row = {"Language": lang}
        
        # LLM averages
        for key in llm_metrics:
            values = llm_metrics[key]
            row[f"LLM_{key}"] = sum(values) / len(values) if values else None
        row["LLM_count"] = len(llm_metrics["entailment_pct"])
        
        # NLI averages
        for key in nli_metrics:
            values = nli_metrics[key]
            row[f"NLI_{key}"] = sum(values) / len(values) if values else None
        row["NLI_count"] = len(nli_metrics["entailment_pct"])
        
        rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "metrics_comparison_by_language.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                formatted_row = {}
                for k, v in row.items():
                    if v is None:
                        formatted_row[k] = "N/A"
                    elif isinstance(v, float):
                        formatted_row[k] = f"{v:.4f}"
                    else:
                        formatted_row[k] = v
                writer.writerow(formatted_row)
        print(f"\nSaved: {output_file}")
    
    return rows


def generate_comparison_by_severity():
    """Generate CSV comparing LLM vs NLI metrics per language-severity."""
    print("\n" + "="*60)
    print("Generating metrics_comparison_by_language_severity.csv")
    print("="*60)
    
    rows = []
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        
        llm_data = load_llm_judge_data(lang)
        nli_data = load_nli_data(lang)
        
        if not llm_data or not nli_data:
            print(f"  Skipping {lang} - missing data")
            continue
        
        # Group by severity
        llm_by_severity = {sev: [] for sev in ALL_SEVERITIES}
        nli_by_severity = {sev: [] for sev in ALL_SEVERITIES}
        
        for llm_row in llm_data:
            severities = llm_row.get("severities", ["Neutral"])
            results = llm_row.get("llm_judge_results", [])
            agg = aggregate_row_metrics(results, "llm")
            if agg:
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        llm_by_severity[sev].append(agg)
        
        for nli_row in nli_data:
            severities = nli_row.get("severities", ["Neutral"])
            results = nli_row.get("nli_results", [])
            agg = aggregate_row_metrics(results, "nli")
            if agg:
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        nli_by_severity[sev].append(agg)
        
        for sev in ALL_SEVERITIES:
            row = {"Language": lang, "Severity": sev}
            
            # LLM metrics for this severity
            llm_list = llm_by_severity[sev]
            if llm_list:
                row["LLM_entailment_pct"] = sum(a["entailment_pct"] for a in llm_list) / len(llm_list)
                row["LLM_neutral_pct"] = sum(a["neutral_pct"] for a in llm_list) / len(llm_list)
                row["LLM_contradiction_pct"] = sum(a["contradiction_pct"] for a in llm_list) / len(llm_list)
                row["LLM_avg_entailment_prob"] = sum(a["avg_entailment_prob"] for a in llm_list) / len(llm_list)
                row["LLM_avg_neutral_prob"] = sum(a["avg_neutral_prob"] for a in llm_list) / len(llm_list)
                row["LLM_avg_contradiction_prob"] = sum(a["avg_contradiction_prob"] for a in llm_list) / len(llm_list)
                row["LLM_count"] = len(llm_list)
            else:
                for key in ["entailment_pct", "neutral_pct", "contradiction_pct", 
                           "avg_entailment_prob", "avg_neutral_prob", "avg_contradiction_prob"]:
                    row[f"LLM_{key}"] = None
                row["LLM_count"] = 0
            
            # NLI metrics for this severity
            nli_list = nli_by_severity[sev]
            if nli_list:
                row["NLI_entailment_pct"] = sum(a["entailment_pct"] for a in nli_list) / len(nli_list)
                row["NLI_neutral_pct"] = sum(a["neutral_pct"] for a in nli_list) / len(nli_list)
                row["NLI_contradiction_pct"] = sum(a["contradiction_pct"] for a in nli_list) / len(nli_list)
                row["NLI_avg_entailment_prob"] = sum(a["avg_entailment_prob"] for a in nli_list) / len(nli_list)
                row["NLI_avg_neutral_prob"] = sum(a["avg_neutral_prob"] for a in nli_list) / len(nli_list)
                row["NLI_avg_contradiction_prob"] = sum(a["avg_contradiction_prob"] for a in nli_list) / len(nli_list)
                row["NLI_count"] = len(nli_list)
            else:
                for key in ["entailment_pct", "neutral_pct", "contradiction_pct",
                           "avg_entailment_prob", "avg_neutral_prob", "avg_contradiction_prob"]:
                    row[f"NLI_{key}"] = None
                row["NLI_count"] = 0
            
            rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "metrics_comparison_by_language_severity.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                formatted_row = {}
                for k, v in row.items():
                    if v is None:
                        formatted_row[k] = "N/A"
                    elif isinstance(v, float):
                        formatted_row[k] = f"{v:.4f}"
                    else:
                        formatted_row[k] = v
                writer.writerow(formatted_row)
        print(f"\nSaved: {output_file}")
    
    return rows


def generate_label_agreement():
    """Generate CSV with global label agreement between LLM and NLI."""
    print("\n" + "="*60)
    print("Generating label_agreement.csv")
    print("="*60)
    
    # Confusion matrix: keys are (llm_label, nli_label)
    confusion = defaultdict(int)
    total_questions = 0
    matches = 0
    
    # Per-label stats
    by_language = {lang: defaultdict(int) for lang in LANGUAGES}
    by_language_total = {lang: 0 for lang in LANGUAGES}
    by_language_matches = {lang: 0 for lang in LANGUAGES}
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        
        llm_data = load_llm_judge_data(lang)
        nli_data = load_nli_data(lang)
        
        if len(llm_data) != len(nli_data):
            print(f"  Warning: LLM ({len(llm_data)}) and NLI ({len(nli_data)}) have different row counts for {lang}")
        
        for llm_row, nli_row in zip(llm_data, nli_data):
            llm_results = llm_row.get("llm_judge_results", [])
            nli_results = nli_row.get("nli_results", [])
            
            # Compare question-by-question
            for llm_r, nli_r in zip(llm_results, nli_results):
                llm_label = normalize_label(llm_r.get("label", ""))
                nli_label = normalize_label(nli_r.get("label", ""))
                
                if llm_label and nli_label:
                    confusion[(llm_label, nli_label)] += 1
                    by_language[lang][(llm_label, nli_label)] += 1
                    total_questions += 1
                    by_language_total[lang] += 1
                    
                    if llm_label == nli_label:
                        matches += 1
                        by_language_matches[lang] += 1
    
    # Write confusion matrix CSV
    output_file = os.path.join(BASE_DIR, "label_agreement.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["", "NLI: ENTAILMENT", "NLI: NEUTRAL", "NLI: CONTRADICTION", "LLM Total"])
        
        for llm_label in LABELS:
            row = [f"LLM: {llm_label}"]
            llm_total = 0
            for nli_label in LABELS:
                count = confusion[(llm_label, nli_label)]
                row.append(count)
                llm_total += count
            row.append(llm_total)
            writer.writerow(row)
        
        # NLI totals
        row = ["NLI Total"]
        for nli_label in LABELS:
            nli_total = sum(confusion[(llm_l, nli_label)] for llm_l in LABELS)
            row.append(nli_total)
        row.append(total_questions)
        writer.writerow(row)
        
        # Summary
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total Questions", total_questions])
        writer.writerow(["Total Matches", matches])
        writer.writerow(["Agreement Rate", f"{matches/total_questions*100:.2f}%" if total_questions > 0 else "N/A"])
        
        # Per-label agreement (diagonal)
        writer.writerow([])
        writer.writerow(["PER-LABEL MATCHES"])
        for label in LABELS:
            count = confusion[(label, label)]
            pct = count / total_questions * 100 if total_questions > 0 else 0
            writer.writerow([label, count, f"{pct:.2f}%"])
        
        # Per-language agreement
        writer.writerow([])
        writer.writerow(["PER-LANGUAGE AGREEMENT"])
        writer.writerow(["Language", "Total", "Matches", "Agreement Rate"])
        for lang in LANGUAGES:
            total = by_language_total[lang]
            match = by_language_matches[lang]
            rate = match / total * 100 if total > 0 else 0
            writer.writerow([lang, total, match, f"{rate:.2f}%"])
    
    print(f"\nSaved: {output_file}")
    print(f"\nGlobal Agreement: {matches}/{total_questions} ({matches/total_questions*100:.2f}%)")
    
    return confusion


def main():
    print("="*60)
    print("Metrics Extension Comparison CSV Generator")
    print("LLM-Judge vs NLI")
    print("="*60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Languages: {LANGUAGES}")
    print(f"Severities: {ALL_SEVERITIES}")
    
    # Generate all CSVs
    generate_comparison_by_language()
    generate_comparison_by_severity()
    generate_label_agreement()
    
    print("\n" + "="*60)
    print("Done! Generated comparison CSVs:")
    print(f"  - {os.path.join(BASE_DIR, 'metrics_comparison_by_language.csv')}")
    print(f"  - {os.path.join(BASE_DIR, 'metrics_comparison_by_language_severity.csv')}")
    print(f"  - {os.path.join(BASE_DIR, 'label_agreement.csv')}")
    print("="*60)


if __name__ == "__main__":
    main()
