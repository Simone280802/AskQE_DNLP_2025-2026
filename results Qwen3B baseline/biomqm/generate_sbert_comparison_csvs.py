"""
Generate SBERT Comparison CSVs: Extensions vs Baseline
Compares NER-extension and Prompt-Ablation strategies against the Baseline for SBERT similarity.
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

# Paths to SBERT data sources
PATHS = {
    "baseline": os.path.join(BASE_DIR, "baseline/evaluation/sbert"),
    "ner": os.path.join(BASE_DIR, "ner-extension/evaluation/sbert"),
    "P1-fewshot": os.path.join(BASE_DIR, "prompt-ablation/QA/P1-fewshot/mapped/evaluation/sbert"),
    "P2-cot": os.path.join(BASE_DIR, "prompt-ablation/QA/P2-cot/mapped/evaluation/sbert"),
    "P3-concise": os.path.join(BASE_DIR, "prompt-ablation/QA/P3-concise/mapped/evaluation/sbert"),
}


def load_baseline_sbert(lang):
    """Load baseline SBERT JSONL file and aggregate similarity by severity."""
    filepath = os.path.join(PATHS["baseline"], f"{lang}-vanilla.jsonl")
    if not os.path.exists(filepath):
        print(f"  Baseline SBERT file not found: {filepath}")
        return {}
    
    stats = {sev: [] for sev in ALL_SEVERITIES}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                scores = row.get('scores', [])
                
                if not scores:
                    continue
                
                # Average sbert_sim for this row
                similarities = [s.get('sbert_sim', 0) for s in scores if 'sbert_sim' in s]
                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    
                    # Distribute to severities (unwind)
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[sev].append(avg_sim)
            except:
                continue
    
    return stats


def load_ner_sbert(lang):
    """Load NER extension SBERT JSONL file and aggregate similarity by severity."""
    filepath = os.path.join(PATHS["ner"], f"{lang}.jsonl")
    if not os.path.exists(filepath):
        print(f"  NER SBERT file not found: {filepath}")
        return {}
    
    stats = {sev: [] for sev in ALL_SEVERITIES}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                
                # NER uses overall_similarity
                sim = row.get('overall_similarity', 0)
                
                for sev in severities:
                    if sev in ALL_SEVERITIES:
                        stats[sev].append(sim)
            except:
                continue
    
    return stats


def load_prompt_ablation_sbert(strategy, lang):
    """Load Prompt Ablation SBERT JSONL file and aggregate similarity by severity."""
    filepath = os.path.join(PATHS[strategy], f"{lang}.jsonl")
    if not os.path.exists(filepath):
        print(f"  {strategy} SBERT file not found: {filepath}")
        return {}
    
    stats = {sev: [] for sev in ALL_SEVERITIES}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                severities = row.get('severities', ['Neutral'])
                scores = row.get('scores', [])
                
                if not scores:
                    continue
                
                # Average sbert_sim for this row
                similarities = [s.get('sbert_sim', 0) for s in scores if 'sbert_sim' in s]
                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats[sev].append(avg_sim)
            except:
                continue
    
    return stats


def aggregate_stats(stats):
    """Calculate averages from list of values."""
    result = {}
    for sev, values in stats.items():
        if values:
            result[sev] = {"sbert_sim": sum(values) / len(values), "count": len(values)}
        else:
            result[sev] = {"sbert_sim": None, "count": 0}
    return result


def aggregate_overall(stats):
    """Aggregate all severities into overall metrics."""
    combined = []
    for sev, values in stats.items():
        combined.extend(values)
    
    if combined:
        return {"sbert_sim": sum(combined) / len(combined), "count": len(combined)}
    else:
        return {"sbert_sim": None, "count": 0}


def generate_sbert_comparison_by_language():
    """Generate CSV comparing SBERT metrics per language across all extensions."""
    print("\n" + "="*60)
    print("Generating sbert_comparison_by_language.csv")
    print("="*60)
    
    rows = []
    extensions = ["baseline", "ner", "P1-fewshot", "P2-cot", "P3-concise"]
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        row = {"Language": lang}
        
        # Load data for all extensions
        data = {}
        data["baseline"] = load_baseline_sbert(lang)
        data["ner"] = load_ner_sbert(lang)
        data["P1-fewshot"] = load_prompt_ablation_sbert("P1-fewshot", lang)
        data["P2-cot"] = load_prompt_ablation_sbert("P2-cot", lang)
        data["P3-concise"] = load_prompt_ablation_sbert("P3-concise", lang)
        
        # Aggregate overall metrics for each extension
        for ext in extensions:
            overall = aggregate_overall(data[ext])
            ext_label = ext.replace("-", "_")
            row[f"{ext_label}_SBERT"] = overall.get("sbert_sim")
            row[f"{ext_label}_count"] = overall.get("count", 0)
        
        # Calculate deltas from baseline
        baseline_sbert = row.get("baseline_SBERT")
        
        for ext in ["ner", "P1-fewshot", "P2-cot", "P3-concise"]:
            ext_label = ext.replace("-", "_")
            ext_sbert = row.get(f"{ext_label}_SBERT")
            
            row[f"delta_{ext_label}_SBERT"] = (ext_sbert - baseline_sbert) if (ext_sbert is not None and baseline_sbert is not None) else None
        
        rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "sbert_comparison_by_language.csv")
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


def generate_sbert_comparison_by_severity():
    """Generate CSV comparing SBERT metrics per language-severity across all extensions."""
    print("\n" + "="*60)
    print("Generating sbert_comparison_by_language_severity.csv")
    print("="*60)
    
    rows = []
    extensions = ["baseline", "ner", "P1-fewshot", "P2-cot", "P3-concise"]
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        
        # Load data for all extensions
        data = {}
        data["baseline"] = aggregate_stats(load_baseline_sbert(lang))
        data["ner"] = aggregate_stats(load_ner_sbert(lang))
        data["P1-fewshot"] = aggregate_stats(load_prompt_ablation_sbert("P1-fewshot", lang))
        data["P2-cot"] = aggregate_stats(load_prompt_ablation_sbert("P2-cot", lang))
        data["P3-concise"] = aggregate_stats(load_prompt_ablation_sbert("P3-concise", lang))
        
        for sev in ALL_SEVERITIES:
            row = {"Language": lang, "Severity": sev}
            
            # Add metrics for each extension
            for ext in extensions:
                ext_label = ext.replace("-", "_")
                sev_data = data[ext].get(sev, {})
                row[f"{ext_label}_SBERT"] = sev_data.get("sbert_sim")
                row[f"{ext_label}_count"] = sev_data.get("count", 0)
            
            # Calculate deltas
            baseline_sbert = row.get("baseline_SBERT")
            
            for ext in ["ner", "P1-fewshot", "P2-cot", "P3-concise"]:
                ext_label = ext.replace("-", "_")
                ext_sbert = row.get(f"{ext_label}_SBERT")
                
                row[f"delta_{ext_label}_SBERT"] = (ext_sbert - baseline_sbert) if (ext_sbert is not None and baseline_sbert is not None) else None
            
            rows.append(row)
    
    # Write CSV
    output_file = os.path.join(BASE_DIR, "sbert_comparison_by_language_severity.csv")
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
    print("SBERT Extension Comparison CSV Generator")
    print("="*60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Languages: {LANGUAGES}")
    print(f"Severities: {ALL_SEVERITIES}")
    
    # Generate both CSVs
    generate_sbert_comparison_by_language()
    generate_sbert_comparison_by_severity()
    
    print("\n" + "="*60)
    print("Done! Generated SBERT comparison CSVs:")
    print(f"  - {os.path.join(BASE_DIR, 'sbert_comparison_by_language.csv')}")
    print(f"  - {os.path.join(BASE_DIR, 'sbert_comparison_by_language_severity.csv')}")
    print("="*60)


if __name__ == "__main__":
    main()
