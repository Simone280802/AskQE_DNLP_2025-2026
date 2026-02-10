"""
Calculate Average Confidence When Label is Assigned

This script calculates the average probability that LLM and NLI assign
to each category ONLY when that category is actually predicted.

Example: When LLM predicts ENTAILMENT, what's the average probability 
it assigned to ENTAILMENT?
"""

import json
import os
from collections import defaultdict

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


def load_data(model_type, lang):
    """Load JSONL file for a specific model and language."""
    if model_type == "llm":
        filepath = os.path.join(RESULTS_DIR, "llm-judge", f"{lang}-llm-judge.jsonl")
        results_key = "llm_judge_results"
    else:
        filepath = os.path.join(RESULTS_DIR, "nli", f"{lang}-nli.jsonl")
        results_key = "nli_results"
    
    if not os.path.exists(filepath):
        return [], results_key
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                data.append(row)
            except json.JSONDecodeError:
                continue
    return data, results_key


def normalize_label(label):
    """Normalize label to uppercase."""
    return label.upper() if label else None


def get_prob_for_label(result, label, model_type):
    """Get the probability for a specific label from a result."""
    probs = result.get("probs", {})
    if model_type == "llm":
        # LLM uses uppercase keys
        return probs.get(label, 0)
    else:
        # NLI uses lowercase keys
        return probs.get(label.lower(), 0)


def calculate_confidence_by_label():
    """Calculate average probability when each label is assigned."""
    
    # Store probabilities: {model: {label: [list of probs]}}
    probs_when_assigned = {
        "LLM": {label: [] for label in LABELS},
        "NLI": {label: [] for label in LABELS}
    }
    
    # Also track by language
    probs_by_language = {
        lang: {
            "LLM": {label: [] for label in LABELS},
            "NLI": {label: [] for label in LABELS}
        }
        for lang in LANGUAGES
    }
    
    for lang in LANGUAGES:
        print(f"Processing {lang}...")
        
        # Process LLM
        llm_data, llm_key = load_data("llm", lang)
        for row in llm_data:
            for result in row.get(llm_key, []):
                label = normalize_label(result.get("label", ""))
                if label in LABELS:
                    prob = get_prob_for_label(result, label, "llm")
                    probs_when_assigned["LLM"][label].append(prob)
                    probs_by_language[lang]["LLM"][label].append(prob)
        
        # Process NLI
        nli_data, nli_key = load_data("nli", lang)
        for row in nli_data:
            for result in row.get(nli_key, []):
                label = normalize_label(result.get("label", ""))
                if label in LABELS:
                    prob = get_prob_for_label(result, label, "nli")
                    probs_when_assigned["NLI"][label].append(prob)
                    probs_by_language[lang]["NLI"][label].append(prob)
    
    # Print results
    print("\n" + "="*70)
    print("AVERAGE PROBABILITY WHEN LABEL IS ASSIGNED (GLOBAL)")
    print("="*70)
    print(f"\n{'Label':<20} {'LLM Avg Prob':<20} {'LLM Count':<15} {'NLI Avg Prob':<20} {'NLI Count':<15}")
    print("-"*70)
    
    for label in LABELS:
        llm_probs = probs_when_assigned["LLM"][label]
        nli_probs = probs_when_assigned["NLI"][label]
        
        llm_avg = sum(llm_probs) / len(llm_probs) if llm_probs else 0
        nli_avg = sum(nli_probs) / len(nli_probs) if nli_probs else 0
        
        print(f"{label:<20} {llm_avg:.4f} ({llm_avg*100:.2f}%)    {len(llm_probs):<15} {nli_avg:.4f} ({nli_avg*100:.2f}%)    {len(nli_probs):<15}")
    
    # Print by language
    print("\n" + "="*70)
    print("AVERAGE PROBABILITY WHEN LABEL IS ASSIGNED (BY LANGUAGE)")
    print("="*70)
    
    for lang in LANGUAGES:
        print(f"\n--- {lang.upper()} ---")
        print(f"{'Label':<20} {'LLM Avg Prob':<25} {'NLI Avg Prob':<25}")
        print("-"*70)
        
        for label in LABELS:
            llm_probs = probs_by_language[lang]["LLM"][label]
            nli_probs = probs_by_language[lang]["NLI"][label]
            
            llm_avg = sum(llm_probs) / len(llm_probs) if llm_probs else 0
            nli_avg = sum(nli_probs) / len(nli_probs) if nli_probs else 0
            
            print(f"{label:<20} {llm_avg:.4f} (n={len(llm_probs):<5})       {nli_avg:.4f} (n={len(nli_probs):<5})")
    
    # Save to CSV
    output_file = os.path.join(BASE_DIR, "confidence_when_assigned.csv")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Global results
        f.write("GLOBAL RESULTS\n")
        f.write("Label,LLM_Avg_Prob,LLM_Count,NLI_Avg_Prob,NLI_Count\n")
        for label in LABELS:
            llm_probs = probs_when_assigned["LLM"][label]
            nli_probs = probs_when_assigned["NLI"][label]
            llm_avg = sum(llm_probs) / len(llm_probs) if llm_probs else 0
            nli_avg = sum(nli_probs) / len(nli_probs) if nli_probs else 0
            f.write(f"{label},{llm_avg:.4f},{len(llm_probs)},{nli_avg:.4f},{len(nli_probs)}\n")
        
        f.write("\n")
        
        # By language
        f.write("BY LANGUAGE\n")
        f.write("Language,Label,LLM_Avg_Prob,LLM_Count,NLI_Avg_Prob,NLI_Count\n")
        for lang in LANGUAGES:
            for label in LABELS:
                llm_probs = probs_by_language[lang]["LLM"][label]
                nli_probs = probs_by_language[lang]["NLI"][label]
                llm_avg = sum(llm_probs) / len(llm_probs) if llm_probs else 0
                nli_avg = sum(nli_probs) / len(nli_probs) if nli_probs else 0
                f.write(f"{lang},{label},{llm_avg:.4f},{len(llm_probs)},{nli_avg:.4f},{len(nli_probs)}\n")
    
    print(f"\n\nResults saved to: {output_file}")
    
    return probs_when_assigned, probs_by_language


if __name__ == "__main__":
    calculate_confidence_by_label()
