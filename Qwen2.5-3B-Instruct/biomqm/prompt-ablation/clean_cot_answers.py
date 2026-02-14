"""
Post-processing script for Chain-of-Thought (CoT) QA files.
Extracts only the FINAL ANSWER from responses that contain REASONING and FINAL ANSWER sections.

Usage:
  python clean_cot_answers.py --input_dir QA/P2-cot --output_dir QA/P2-cot/clean
  python clean_cot_answers.py --input_file QA/P2-cot/source-P2-cot.jsonl --output_file QA/P2-cot/clean/source-P2-cot.jsonl
"""

import json
import os
import re
import argparse
from pathlib import Path


def extract_final_answer(text):
    """
    Extract the FINAL ANSWER from a CoT response.
    
    Handles formats like:
    - "REASONING: ... FINAL ANSWER: ..."
    - "REASONING : ... FINAL ANSWER : ..."
    - Just "FINAL ANSWER: ..." 
    """
    if not text or not isinstance(text, str):
        return text
    
    # Pattern to match FINAL ANSWER (with optional space before colon)
    patterns = [
        r'FINAL\s*ANSWER\s*:\s*(.+?)$',  # FINAL ANSWER: ... to end of string
        r'FINAL\s*ANSWER\s*:\s*(.+)',     # Fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up any trailing whitespace or newlines
            return answer.strip()
    
    # If no FINAL ANSWER pattern found, return original text
    return text


def clean_answers_list(answers):
    """Clean a list of answers by extracting FINAL ANSWER from each."""
    if not answers:
        return answers
    
    if isinstance(answers, str):
        return extract_final_answer(answers)
    
    if isinstance(answers, list):
        return [extract_final_answer(str(a)) if a else a for a in answers]
    
    return answers


def process_file(input_path, output_path):
    """Process a single JSONL file and clean the answers."""
    cleaned_count = 0
    total_count = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                row = json.loads(line)
                total_count += 1
                
                # Clean answers field
                if 'answers' in row:
                    original = row['answers']
                    row['answers'] = clean_answers_list(row['answers'])
                    if original != row['answers']:
                        cleaned_count += 1
                
                f_out.write(json.dumps(row, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                continue
    
    print(f"Processed {input_path}:")
    print(f"  Total rows: {total_count}")
    print(f"  Cleaned rows: {cleaned_count}")
    
    return total_count, cleaned_count


def process_directory(input_dir, output_dir):
    """Process all JSONL files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    os.makedirs(output_path, exist_ok=True)
    
    total_files = 0
    total_rows = 0
    total_cleaned = 0
    
    for jsonl_file in input_path.glob('*.jsonl'):
        output_file = output_path / f"clean-{jsonl_file.name}"
        rows, cleaned = process_file(str(jsonl_file), str(output_file))
        total_files += 1
        total_rows += rows
        total_cleaned += cleaned
    
    print(f"\n{'='*50}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*50}")
    print(f"Files processed: {total_files}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with cleaned answers: {total_cleaned}")
    print(f"Output directory: {output_path}")
    
    return total_files, total_rows, total_cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean CoT answers by extracting FINAL ANSWER")
    parser.add_argument("--input_dir", type=str, help="Directory containing QA JSONL files")
    parser.add_argument("--output_dir", type=str, help="Output directory for cleaned files")
    parser.add_argument("--input_file", type=str, help="Single input file to process")
    parser.add_argument("--output_file", type=str, help="Output file for single file processing")
    
    args = parser.parse_args()
    
    if args.input_file and args.output_file:
        process_file(args.input_file, args.output_file)
    elif args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir)
    else:
        print("Please provide either --input_dir and --output_dir, or --input_file and --output_file")


if __name__ == "__main__":
    main()
