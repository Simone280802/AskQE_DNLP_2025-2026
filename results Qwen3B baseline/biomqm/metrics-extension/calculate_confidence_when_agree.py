"""
Calculate Average Confidence When BOTH Models Agree on Label

This script calculates the average probability that LLM and NLI assign
to each category when BOTH models predict the SAME label.

Example: When both LLM and NLI predict ENTAILMENT, what's the average 
probability each model assigned to ENTAILMENT?
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
        return probs.get(label, 0)
    else:
        return probs.get(label.lower(), 0)


def calculate_confidence_when_agree():
    """Calculate average probability when both models agree on label."""
    
    # Store probabilities when both agree: {label: {"llm": [], "nli": []}}
    probs_when_agree = {
        label: {"llm": [], "nli": []}
        for label in LABELS
    }
    
    # Also track by language
    probs_by_language = {
        lang: {
            label: {"llm": [], "nli": []}
            for label in LABELS
        }
        for lang in LANGUAGES
    }
    
    # Count agreements and disagreements
    total_comparisons = 0
    agreements = 0
    
    for lang in LANGUAGES:
        print(f"Processing {lang}...")
        
        # Load both models
        llm_data, llm_key = load_data("llm", lang)
        nli_data, nli_key = load_data("nli", lang)
        
        # Iterate through aligned rows
        for llm_row, nli_row in zip(llm_data, nli_data):
            llm_results = llm_row.get(llm_key, [])
            nli_results = nli_row.get(nli_key, [])
            
            # Compare question by question
            for llm_r, nli_r in zip(llm_results, nli_results):
                llm_label = normalize_label(llm_r.get("label", ""))
                nli_label = normalize_label(nli_r.get("label", ""))
                
                if not llm_label or not nli_label:
                    continue
                
                total_comparisons += 1
                
                # Check if they agree
                if llm_label == nli_label:
                    agreements += 1
                    agreed_label = llm_label
                    
                    # Get probabilities for the agreed label
                    llm_prob = get_prob_for_label(llm_r, agreed_label, "llm")
                    nli_prob = get_prob_for_label(nli_r, agreed_label, "nli")
                    
                    probs_when_agree[agreed_label]["llm"].append(llm_prob)
                    probs_when_agree[agreed_label]["nli"].append(nli_prob)
                    
                    probs_by_language[lang][agreed_label]["llm"].append(llm_prob)
                    probs_by_language[lang][agreed_label]["nli"].append(nli_prob)
    
    # Print results
    print("\n" + "="*80)
    print("AVERAGE PROBABILITY WHEN BOTH MODELS AGREE ON LABEL (GLOBAL)")
    print("="*80)
    print(f"\nTotal comparisons: {total_comparisons}")
    print(f"Total agreements: {agreements} ({agreements/total_comparisons*100:.2f}%)")
    
    print(f"\n{'Label':<20} {'LLM Avg Prob':<20} {'NLI Avg Prob':<20} {'Count':<15}")
    print("-"*80)
    
    for label in LABELS:
        llm_probs = probs_when_agree[label]["llm"]
        nli_probs = probs_when_agree[label]["nli"]
        
        if llm_probs:
            llm_avg = sum(llm_probs) / len(llm_probs)
            nli_avg = sum(nli_probs) / len(nli_probs)
            count = len(llm_probs)
            print(f"{label:<20} {llm_avg:.4f} ({llm_avg*100:.2f}%)    {nli_avg:.4f} ({nli_avg*100:.2f}%)    {count:<15}")
        else:
            print(f"{label:<20} N/A                  N/A                  0")
    
    # Print by language
    print("\n" + "="*80)
    print("AVERAGE PROBABILITY WHEN BOTH AGREE (BY LANGUAGE)")
    print("="*80)
    
    for lang in LANGUAGES:
        print(f"\n--- {lang.upper()} ---")
        print(f"{'Label':<20} {'LLM Avg Prob':<25} {'NLI Avg Prob':<25} {'Count':<10}")
        print("-"*80)
        
        for label in LABELS:
            llm_probs = probs_by_language[lang][label]["llm"]
            nli_probs = probs_by_language[lang][label]["nli"]
            
            if llm_probs:
                llm_avg = sum(llm_probs) / len(llm_probs)
                nli_avg = sum(nli_probs) / len(nli_probs)
                count = len(llm_probs)
                print(f"{label:<20} {llm_avg:.4f} ({llm_avg*100:.2f}%)       {nli_avg:.4f} ({nli_avg*100:.2f}%)       {count}")
            else:
                print(f"{label:<20} N/A                       N/A                       0")
    
    # Save to CSV
    output_file = os.path.join(BASE_DIR, "confidence_when_both_agree.csv")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SUMMARY\n")
        f.write(f"Total Comparisons,{total_comparisons}\n")
        f.write(f"Total Agreements,{agreements}\n")
        f.write(f"Agreement Rate,{agreements/total_comparisons*100:.2f}%\n")
        f.write("\n")
        
        # Global results
        f.write("GLOBAL RESULTS (WHEN BOTH AGREE)\n")
        f.write("Label,LLM_Avg_Prob,NLI_Avg_Prob,Count\n")
        for label in LABELS:
            llm_probs = probs_when_agree[label]["llm"]
            nli_probs = probs_when_agree[label]["nli"]
            if llm_probs:
                llm_avg = sum(llm_probs) / len(llm_probs)
                nli_avg = sum(nli_probs) / len(nli_probs)
                f.write(f"{label},{llm_avg:.4f},{nli_avg:.4f},{len(llm_probs)}\n")
            else:
                f.write(f"{label},N/A,N/A,0\n")
        
        f.write("\n")
        
        # By language
        f.write("BY LANGUAGE (WHEN BOTH AGREE)\n")
        f.write("Language,Label,LLM_Avg_Prob,NLI_Avg_Prob,Count\n")
        for lang in LANGUAGES:
            for label in LABELS:
                llm_probs = probs_by_language[lang][label]["llm"]
                nli_probs = probs_by_language[lang][label]["nli"]
                if llm_probs:
                    llm_avg = sum(llm_probs) / len(llm_probs)
                    nli_avg = sum(nli_probs) / len(nli_probs)
                    f.write(f"{lang},{label},{llm_avg:.4f},{nli_avg:.4f},{len(llm_probs)}\n")
                else:
                    f.write(f"{lang},{label},N/A,N/A,0\n")
    
    print(f"\n\nResults saved to: {output_file}")
    
    return probs_when_agree, probs_by_language


if __name__ == "__main__":
    calculate_confidence_when_agree()
