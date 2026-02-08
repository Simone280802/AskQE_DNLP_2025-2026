"""
Add BLEU and chrF metrics to NER extension evaluation CSVs.
Reads JSONL files and regenerates CSVs with all metrics.
"""

import json
import os
import csv
import sys

# Add utils path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import sacrebleu
except ImportError:
    print("Installing sacrebleu...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sacrebleu'], check=True)
    import sacrebleu

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SC_DIR = os.path.join(BASE_DIR, "string-comparison")
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def bleu_score(prediction, reference):
    """Compute sentence-level BLEU score."""
    if not prediction or not reference:
        return 0.0
    try:
        return sacrebleu.sentence_bleu(prediction, [reference]).score
    except:
        return 0.0


def chrf_score(prediction, reference):
    """Compute sentence-level chrF score."""
    if not prediction or not reference:
        return 0.0
    try:
        return sacrebleu.sentence_chrf(prediction, [reference]).score
    except:
        return 0.0


def process_all_jsonl_files():
    """Process all JSONL files and extract scores with all 4 metrics."""
    all_rows = []
    stats_by_lang = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in LANGUAGES}
    
    for lang in LANGUAGES:
        jsonl_file = os.path.join(SC_DIR, f"{lang}.jsonl")
        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}")
            continue
        
        print(f"Processing {lang}...")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    severities = row.get('severities', ['Neutral'])
                    entity_metrics = row.get('entity_metrics', {})
                    
                    # Get existing F1 and EM
                    overall_f1 = row.get('overall_f1', 0)
                    overall_em = row.get('overall_em', 0)
                    
                    # Calculate BLEU and chrF from entity answers
                    bleu_scores = []
                    chrf_scores = []
                    
                    for entity_type, metrics in entity_metrics.items():
                        answer_src = metrics.get('answer_src', '')
                        answer_bt = metrics.get('answer_bt', '')
                        
                        if answer_src and answer_bt:
                            bleu_scores.append(bleu_score(answer_bt, answer_src))
                            chrf_scores.append(chrf_score(answer_bt, answer_src))
                    
                    # Average BLEU and chrF
                    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
                    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0
                    
                    # Add to per-severity stats (unwind)
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats_by_lang[lang][sev].append((overall_f1, overall_em, avg_chrf, avg_bleu))
                            
                            # Add to all_rows
                            all_rows.append({
                                'lang': lang,
                                'severity': sev,
                                'f1': overall_f1,
                                'em': overall_em,
                                'chrf': avg_chrf,
                                'bleu': avg_bleu
                            })
                    
                except json.JSONDecodeError:
                    continue
    
    return all_rows, stats_by_lang


def save_csvs(all_rows, stats_by_lang):
    """Save the updated CSVs with BLEU and chrF."""
    
    # 1. string_comparison_all_languages.csv
    all_csv = os.path.join(BASE_DIR, "string_comparison_all_languages.csv")
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['lang', 'severity', 'f1', 'em', 'chrf', 'bleu'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Saved: {all_csv}")
    
    # 2. string_comparison_summary_by_lang.csv
    summary_csv = os.path.join(BASE_DIR, "string_comparison_summary_by_lang.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lang', 'avg_f1', 'avg_em', 'avg_chrf', 'avg_bleu'])
        
        for lang in LANGUAGES:
            all_scores = []
            for sev in ALL_SEVERITIES:
                all_scores.extend(stats_by_lang[lang][sev])
            
            if all_scores:
                avg_f1 = sum(s[0] for s in all_scores) / len(all_scores)
                avg_em = sum(s[1] for s in all_scores) / len(all_scores)
                avg_chrf = sum(s[2] for s in all_scores) / len(all_scores)
                avg_bleu = sum(s[3] for s in all_scores) / len(all_scores)
                writer.writerow([lang, avg_f1, avg_em, avg_chrf, avg_bleu])
    print(f"Saved: {summary_csv}")
    
    # 3. string_comparison_summary_by_lang_severity.csv
    severity_csv = os.path.join(BASE_DIR, "string_comparison_summary_by_lang_severity.csv")
    with open(severity_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lang', 'severity', 'count', 'avg_f1', 'avg_em', 'avg_chrf', 'avg_bleu'])
        
        for lang in LANGUAGES:
            for sev in ALL_SEVERITIES:
                scores = stats_by_lang[lang][sev]
                if scores:
                    count = len(scores)
                    avg_f1 = sum(s[0] for s in scores) / count
                    avg_em = sum(s[1] for s in scores) / count
                    avg_chrf = sum(s[2] for s in scores) / count
                    avg_bleu = sum(s[3] for s in scores) / count
                    writer.writerow([lang, sev, count, avg_f1, avg_em, avg_chrf, avg_bleu])
    print(f"Saved: {severity_csv}")


def main():
    print("=== Adding BLEU and chrF to NER Extension CSVs ===\n")
    
    all_rows, stats_by_lang = process_all_jsonl_files()
    
    if all_rows:
        save_csvs(all_rows, stats_by_lang)
        print("\n=== Done! ===")
    else:
        print("No data found in JSONL files.")


if __name__ == "__main__":
    main()
