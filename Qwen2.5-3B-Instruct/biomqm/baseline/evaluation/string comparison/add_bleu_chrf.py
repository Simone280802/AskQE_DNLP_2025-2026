"""
Add BLEU and chrF metrics to baseline evaluation CSVs.
Reads existing JSONL files and regenerates CSVs with all 4 metrics.
"""

import json
import os
import csv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SC_DIR = os.path.join(BASE_DIR, "string comparison")
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
ALL_SEVERITIES = ["Neutral", "Minor", "Major", "Critical"]


def parse_json_field(field):
    """Parse a field that may be stored as JSON string or already as list."""
    if isinstance(field, list):
        return field
    if not field or (isinstance(field, str) and field.strip() == ""):
        return []
    try:
        parsed = json.loads(field)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        return [field]


def process_all_jsonl_files():
    """Process all JSONL files and extract scores with all 4 metrics."""
    all_rows = []
    stats_by_lang = {lang: {sev: [] for sev in ALL_SEVERITIES} for lang in LANGUAGES}
    
    for lang in LANGUAGES:
        jsonl_file = os.path.join(SC_DIR, f"{lang}-vanilla.jsonl")
        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}")
            continue
        
        print(f"Processing {lang}...")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    severities = row.get('severities', ['Neutral'])
                    scores_list = row.get('scores', [])
                    
                    if not scores_list:
                        continue
                    
                    # Calculate average scores for this row
                    n = len(scores_list)
                    avg_f1 = sum(s.get('f1', 0) for s in scores_list) / n
                    avg_em = sum(1 if s.get('em', False) else 0 for s in scores_list) / n
                    avg_chrf = sum(s.get('chrf', 0) for s in scores_list) / n
                    avg_bleu = sum(s.get('bleu', 0) for s in scores_list) / n
                    
                    # Add to per-severity stats (unwind)
                    for sev in severities:
                        if sev in ALL_SEVERITIES:
                            stats_by_lang[lang][sev].append((avg_f1, avg_em, avg_chrf, avg_bleu))
                    
                    # Add individual score rows for all_languages.csv
                    for score in scores_list:
                        for sev in severities:
                            all_rows.append({
                                'lang': lang,
                                'severity': sev,
                                'f1': score.get('f1', 0),
                                'em': score.get('em', False),
                                'chrf': score.get('chrf', 0),
                                'bleu': score.get('bleu', 0)
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
    print("=== Adding BLEU and chrF to CSVs ===\n")
    
    all_rows, stats_by_lang = process_all_jsonl_files()
    
    if all_rows:
        save_csvs(all_rows, stats_by_lang)
        print("\n=== Done! ===")
    else:
        print("No data found in JSONL files.")


if __name__ == "__main__":
    main()
