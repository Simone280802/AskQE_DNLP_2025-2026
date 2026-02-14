"""
Add BLEU and chrF metrics to NER extension evaluation JSONL files.
Calculates metrics for each entity answer pair and adds overall_bleu/overall_chrf.
"""

import json
import os
from sacrebleu.metrics import BLEU, CHRF

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def calculate_bleu(prediction, reference):
    """Calculate sentence-level BLEU score."""
    if not prediction or not reference:
        return 0.0
    if prediction == "[NOT FOUND]" or reference == "[NOT FOUND]":
        return 0.0
    try:
        metric = BLEU(effective_order=True)
        return metric.sentence_score(prediction, [reference]).score
    except:
        return 0.0


def calculate_chrf(prediction, reference):
    """Calculate sentence-level chrF score."""
    if not prediction or not reference:
        return 0.0
    if prediction == "[NOT FOUND]" or reference == "[NOT FOUND]":
        return 0.0
    try:
        metric = CHRF()
        return metric.sentence_score(prediction, [reference]).score
    except:
        return 0.0


def process_file(lang):
    """Process a single language JSONL file and add BLEU/chrF metrics."""
    input_file = os.path.join(BASE_DIR, f"{lang}.jsonl")
    output_file = os.path.join(BASE_DIR, f"{lang}_with_bleu_chrf.jsonl")
    
    if not os.path.exists(input_file):
        print(f"  File not found: {input_file}")
        return 0
    
    processed_count = 0
    updated_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line.strip())
                entity_metrics = row.get('entity_metrics', {})
                
                # Calculate BLEU and chrF for each entity
                bleu_scores = []
                chrf_scores = []
                
                for entity_type, metrics in entity_metrics.items():
                    answer_src = metrics.get('answer_src', '')
                    answer_bt = metrics.get('answer_bt', '')
                    
                    # Calculate metrics (prediction=answer_bt, reference=answer_src)
                    bleu = calculate_bleu(answer_bt, answer_src)
                    chrf = calculate_chrf(answer_bt, answer_src)
                    
                    # Add to entity metrics
                    metrics['bleu'] = bleu
                    metrics['chrf'] = chrf
                    
                    bleu_scores.append(bleu)
                    chrf_scores.append(chrf)
                
                # Calculate overall BLEU and chrF
                if bleu_scores:
                    row['overall_bleu'] = sum(bleu_scores) / len(bleu_scores)
                    row['overall_chrf'] = sum(chrf_scores) / len(chrf_scores)
                else:
                    row['overall_bleu'] = 0.0
                    row['overall_chrf'] = 0.0
                
                updated_rows.append(row)
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing line {line_num}: {e}")
                continue
    
    # Write updated file
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in updated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"  Processed {processed_count} rows -> {output_file}")
    return processed_count


def replace_original_files():
    """Replace original files with updated ones."""
    for lang in LANGUAGES:
        original = os.path.join(BASE_DIR, f"{lang}.jsonl")
        updated = os.path.join(BASE_DIR, f"{lang}_with_bleu_chrf.jsonl")
        backup = os.path.join(BASE_DIR, f"{lang}_backup.jsonl")
        
        if os.path.exists(updated):
            # Backup original
            if os.path.exists(original):
                os.rename(original, backup)
            # Replace with updated
            os.rename(updated, original)
            print(f"  Replaced {lang}.jsonl (backup: {lang}_backup.jsonl)")


def main():
    print("=" * 60)
    print("Adding BLEU and chrF to NER Extension JSONL files")
    print("=" * 60)
    
    total_processed = 0
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang}...")
        count = process_file(lang)
        total_processed += count
    
    print(f"\n{'=' * 60}")
    print(f"Total rows processed: {total_processed}")
    
    # Ask before replacing
    print("\nReplacing original files with updated versions...")
    replace_original_files()
    
    print("\n" + "=" * 60)
    print("Done! BLEU and chrF metrics have been added.")
    print("=" * 60)


if __name__ == "__main__":
    main()
