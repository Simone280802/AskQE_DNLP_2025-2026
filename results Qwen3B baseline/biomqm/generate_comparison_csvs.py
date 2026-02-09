"""
Generate Comparison CSVs: Extensions vs Baseline
Compares NER-extension and Prompt-Ablation strategies against the Baseline.
Outputs two CSVs: by language and by language-severity pair.
"""

import json
import os
import csv
from collections import defaultdict

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]

# Paths to data sources
PATHS = {
    "baseline": os.path.join(BASE_DIR, "baseline/evaluation/string comparison"),
    "ner": os.path.join(BASE_DIR, "ner-extension/evaluation/string-comparison"),
    "P1-fewshot": os.path.join(BASE_DIR, "prompt-ablation/QA/P1-fewshot/mapped/evaluation/string-comparison"),
    "P2-cot": os.path.join(BASE_DIR, "prompt-ablation/QA/P2-cot/mapped/evaluation/string-comparison"),
    "P3-concise": os.path.join(BASE_DIR, "prompt-ablation/QA/P3-concise/mapped/evaluation/string-comparison"),
}


def load_baseline_data(lang):
    """Load baseline JSONL file and aggregate metrics by severity."""
    filepath = os.path.join(PATHS["baseline"], f"{lang}-vanilla.jsonl")
    if not os.path.exists(filepath):
        print(f"  Baseline file not found: {filepath}")
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
                
                # Average scores for this row
                n = len(scores)
                avg_f1 = sum(s.get('f1', 0) for s in scores) / n
                avg_em = sum(1 if s.get('em', False) else 0 for s in scores) / n
                avg_chrf = sum(s.get('chrf', 0) for s in scores) / n
                avg_bleu = sum(s.get('bleu', 0) for s in scores) / n
                
                # Distribute to severities (unwind)
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        stats[sev]["f1"].append(avg_f1)
                        stats[sev]["em"].append(avg_em)
                        stats[sev]["chrf"].append(avg_chrf)
                        stats[sev]["bleu"].append(avg_bleu)
            except:
                continue
    
    return stats


def load_ner_data(lang):
    """Load NER extension JSONL file and aggregate metrics by severity."""
    filepath = os.path.join(PATHS["ner"], f"{lang}.jsonl")
    if not os.path.exists(filepath):
        print(f"  NER file not found: {filepath}")
        return {}
    
    stats = {sev: {"f1": [], "em": [], "chrf": [], "bleu": []} for sev in ALL_SEVERITIES}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                
                f1 = row.get('overall_f1', 0)
                em = row.get('overall_em', 0)
                chrf = row.get('overall_chrf', 0)
                bleu = row.get('overall_bleu', 0)
                
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        stats[sev]["f1"].append(f1)
                        stats[sev]["em"].append(em)
                        stats[sev]["chrf"].append(chrf)
                        stats[sev]["bleu"].append(bleu)
            except:
                continue
    
    return stats


def load_prompt_ablation_data(strategy, lang):
    """Load Prompt Ablation JSONL file and aggregate metrics by severity."""
    filepath = os.path.join(PATHS[strategy], f"{lang}.jsonl")
    if not os.path.exists(filepath):
        print(f"  {strategy} file not found: {filepath}")
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


def aggregate_stats(stats):
    """Calculate averages from list of values."""
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
    combined = {"f1": [], "em": [], "chrf": [], "bleu": []}
    for sev, metrics in stats.items():
        for metric_name in ["f1", "em", "chrf", "bleu"]:
            combined[metric_name].extend(metrics.get(metric_name, []))
    
    result = {}
    for metric_name, values in combined.items():
        if values:
            result[metric_name] = sum(values) / len(values)
            result[f"{metric_name}_count"] = len(values)
        else:
            result[metric_name] = None
            result[f"{metric_name}_count"] = 0
    return result


def generate_comparison_by_language():
    """Generate CSV comparing metrics per language across all extensions."""
    print("\n" + "="*60)
    print("Generating comparison_by_language.csv")
    print("="*60)
    
    rows = []
    extensions = ["baseline", "ner", "P1-fewshot", "P2-cot", "P3-concise"]
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        row = {"Language": lang}
        
        # Load data for all extensions
        data = {}
        data["baseline"] = load_baseline_data(lang)
        data["ner"] = load_ner_data(lang)
        data["P1-fewshot"] = load_prompt_ablation_data("P1-fewshot", lang)
        data["P2-cot"] = load_prompt_ablation_data("P2-cot", lang)
        data["P3-concise"] = load_prompt_ablation_data("P3-concise", lang)
        
        # Aggregate overall metrics for each extension
        for ext in extensions:
            overall = aggregate_overall(data[ext])
            ext_label = ext.replace("-", "_")
            row[f"{ext_label}_F1"] = overall.get("f1")
            row[f"{ext_label}_EM"] = overall.get("em")
            row[f"{ext_label}_chrF"] = overall.get("chrf")
            row[f"{ext_label}_BLEU"] = overall.get("bleu")
            row[f"{ext_label}_count"] = overall.get("f1_count", 0)
        
        # Calculate deltas from baseline
        baseline_f1 = row.get("baseline_F1")
        baseline_em = row.get("baseline_EM")
        baseline_chrf = row.get("baseline_chrF")
        baseline_bleu = row.get("baseline_BLEU")
        
        for ext in ["ner", "P1-fewshot", "P2-cot", "P3-concise"]:
            ext_label = ext.replace("-", "_")
            ext_f1 = row.get(f"{ext_label}_F1")
            ext_em = row.get(f"{ext_label}_EM")
            ext_chrf = row.get(f"{ext_label}_chrF")
            ext_bleu = row.get(f"{ext_label}_BLEU")
            
            row[f"delta_{ext_label}_F1"] = (ext_f1 - baseline_f1) if (ext_f1 is not None and baseline_f1 is not None) else None
            row[f"delta_{ext_label}_EM"] = (ext_em - baseline_em) if (ext_em is not None and baseline_em is not None) else None
            row[f"delta_{ext_label}_chrF"] = (ext_chrf - baseline_chrf) if (ext_chrf is not None and baseline_chrf is not None) else None
            row[f"delta_{ext_label}_BLEU"] = (ext_bleu - baseline_bleu) if (ext_bleu is not None and baseline_bleu is not None) else None
        
        rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "comparison_by_language.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Format numeric values
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
    """Generate CSV comparing metrics per language-severity across all extensions."""
    print("\n" + "="*60)
    print("Generating comparison_by_language_severity.csv")
    print("="*60)
    
    rows = []
    extensions = ["baseline", "ner", "P1-fewshot", "P2-cot", "P3-concise"]
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        
        # Load data for all extensions
        data = {}
        data["baseline"] = aggregate_stats(load_baseline_data(lang))
        data["ner"] = aggregate_stats(load_ner_data(lang))
        data["P1-fewshot"] = aggregate_stats(load_prompt_ablation_data("P1-fewshot", lang))
        data["P2-cot"] = aggregate_stats(load_prompt_ablation_data("P2-cot", lang))
        data["P3-concise"] = aggregate_stats(load_prompt_ablation_data("P3-concise", lang))
        
        for sev in ALL_SEVERITIES:
            row = {"Language": lang, "Severity": sev}
            
            # Add metrics for each extension
            for ext in extensions:
                ext_label = ext.replace("-", "_")
                sev_data = data[ext].get(sev, {})
                row[f"{ext_label}_F1"] = sev_data.get("f1")
                row[f"{ext_label}_EM"] = sev_data.get("em")
                row[f"{ext_label}_chrF"] = sev_data.get("chrf")
                row[f"{ext_label}_BLEU"] = sev_data.get("bleu")
                row[f"{ext_label}_count"] = sev_data.get("f1_count", 0)
            
            # Calculate deltas
            baseline_f1 = row.get("baseline_F1")
            baseline_em = row.get("baseline_EM")
            baseline_chrf = row.get("baseline_chrF")
            baseline_bleu = row.get("baseline_BLEU")
            
            for ext in ["ner", "P1-fewshot", "P2-cot", "P3-concise"]:
                ext_label = ext.replace("-", "_")
                ext_f1 = row.get(f"{ext_label}_F1")
                ext_em = row.get(f"{ext_label}_EM")
                ext_chrf = row.get(f"{ext_label}_chrF")
                ext_bleu = row.get(f"{ext_label}_BLEU")
                
                row[f"delta_{ext_label}_F1"] = (ext_f1 - baseline_f1) if (ext_f1 is not None and baseline_f1 is not None) else None
                row[f"delta_{ext_label}_EM"] = (ext_em - baseline_em) if (ext_em is not None and baseline_em is not None) else None
                row[f"delta_{ext_label}_chrF"] = (ext_chrf - baseline_chrf) if (ext_chrf is not None and baseline_chrf is not None) else None
                row[f"delta_{ext_label}_BLEU"] = (ext_bleu - baseline_bleu) if (ext_bleu is not None and baseline_bleu is not None) else None
            
            rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "comparison_by_language_severity.csv")
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


def main():
    print("="*60)
    print("Extension Comparison CSV Generator")
    print("="*60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Languages: {LANGUAGES}")
    print(f"Severities: {ALL_SEVERITIES}")
    
    # Generate both CSVs
    generate_comparison_by_language()
    generate_comparison_by_severity()
    
    print("\n" + "="*60)
    print("Done! Generated comparison CSVs:")
    print(f"  - {os.path.join(BASE_DIR, 'comparison_by_language.csv')}")
    print(f"  - {os.path.join(BASE_DIR, 'comparison_by_language_severity.csv')}")
    print("="*60)


if __name__ == "__main__":
    main()
