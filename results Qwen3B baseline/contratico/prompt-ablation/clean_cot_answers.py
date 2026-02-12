"""
Post-processing script for Chain-of-Thought (CoT) QA files (contraTICO).
Removes REASONING sections and keeps only the FINAL ANSWER content.

Cases handled:
  1. "REASONING : ... FINAL ANSWER: <answer>"  →  "<answer>"
  2. "REASONING : ... FINAL ANSWER : <answer>"  →  "<answer>"  (space before colon)
  3. "REASONING : ... (no FINAL ANSWER)"          →  "" (empty string)
  4. "FINAL ANSWER: <answer>" (no REASONING)      →  "<answer>"
  5. Text without REASONING or FINAL ANSWER       →  kept as-is

Usage:
  # Process all files in QA/P2-cot (bt + source), write cleaned versions in-place:
  python clean_cot_answers.py

  # Or specify custom input/output dirs:
  python clean_cot_answers.py --input_dir QA/P2-cot --output_dir QA/P2-cot-clean
"""

import json
import os
import re
import argparse
from pathlib import Path


def extract_final_answer(text):
    """
    Extract the FINAL ANSWER from a CoT response.
    If there's a FINAL ANSWER, return only that part.
    If there's only REASONING with no FINAL ANSWER, return empty string.
    If there's no REASONING pattern at all, return text as-is.
    """
    if not text or not isinstance(text, str):
        return text

    text_stripped = text.strip()

    # Check if text contains REASONING pattern (both "REASONING :" and "[REASONING]" formats)
    has_reasoning = bool(re.search(r'(?:\[?\s*REASONING\s*\]?\s*:?)', text_stripped, re.IGNORECASE))

    # Try to extract FINAL ANSWER with colon format: "FINAL ANSWER: ..."
    final_answer_matches = list(re.finditer(
        r'FINAL\s*ANSWER\s*:\s*(.*)',
        text_stripped,
        re.IGNORECASE | re.DOTALL
    ))

    if final_answer_matches:
        # Take the last FINAL ANSWER match (handles nested cases)
        answer = final_answer_matches[-1].group(1).strip()
        if answer:
            return answer
        # FINAL ANSWER: with empty content → return empty
        if has_reasoning:
            return ""

    # Try bracket format: "[FINAL ANSWER] ..."
    bracket_matches = list(re.finditer(
        r'\[FINAL\s*ANSWER\]\s*(.*)',
        text_stripped,
        re.IGNORECASE | re.DOTALL
    ))

    if bracket_matches:
        answer = bracket_matches[-1].group(1).strip()
        if answer:
            return answer
        if has_reasoning:
            return ""

    # If we found REASONING but no valid FINAL ANSWER, return empty string
    if has_reasoning:
        return ""

    # No REASONING pattern found, return original text unchanged
    return text_stripped


def clean_answers_list(answers):
    """Clean a list of answers by extracting FINAL ANSWER from each."""
    if not answers:
        return answers

    if isinstance(answers, str):
        return extract_final_answer(answers)

    if isinstance(answers, list):
        cleaned = []
        for a in answers:
            if a and isinstance(a, str):
                cleaned.append(extract_final_answer(a))
            else:
                cleaned.append(a)
        return cleaned

    return answers


def process_file(input_path, output_path):
    """Process a single JSONL file and clean the answers."""
    cleaned_count = 0
    total_count = 0
    emptied_count = 0

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                total_count += 1

                # Clean 'answers' field (contraTICO format)
                if 'answers' in row:
                    original = row['answers']
                    row['answers'] = clean_answers_list(row['answers'])
                    if original != row['answers']:
                        cleaned_count += 1
                        # Count how many answers became empty
                        if isinstance(row['answers'], list):
                            emptied_count += sum(1 for a in row['answers'] if a == "")

                # Also clean 'answers_src' and 'answers_bt' fields (biomqm format, just in case)
                for field in ['answers_src', 'answers_bt']:
                    if field in row:
                        original = row[field]
                        row[field] = clean_answers_list(row[field])
                        if original != row[field]:
                            cleaned_count += 1

                rows.append(row)

            except json.JSONDecodeError:
                continue

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"  {os.path.basename(input_path)}: {total_count} rows, {cleaned_count} cleaned, {emptied_count} emptied (REASONING without FINAL ANSWER)")

    return total_count, cleaned_count, emptied_count


def process_directory(input_dir, output_dir, in_place=False):
    """Recursively process all JSONL files in a directory."""
    input_path = Path(input_dir)

    total_files = 0
    total_rows = 0
    total_cleaned = 0
    total_emptied = 0

    for jsonl_file in sorted(input_path.rglob('*.jsonl')):
        rel_path = jsonl_file.relative_to(input_path)

        if in_place:
            out_file = jsonl_file
        else:
            out_file = Path(output_dir) / rel_path
            os.makedirs(out_file.parent, exist_ok=True)

        rows, cleaned, emptied = process_file(str(jsonl_file), str(out_file))
        total_files += 1
        total_rows += rows
        total_cleaned += cleaned
        total_emptied += emptied

    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {total_files}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with cleaned answers: {total_cleaned}")
    print(f"Answers emptied (REASONING without FINAL ANSWER): {total_emptied}")
    if not in_place:
        print(f"Output directory: {output_dir}")
    else:
        print(f"Files modified in-place: {input_dir}")

    return total_files, total_rows, total_cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean CoT answers by extracting FINAL ANSWER")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing QA JSONL files (default: QA/P2-cot)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for cleaned files (default: overwrite in-place)")

    args = parser.parse_args()

    # Default: process QA/P2-cot in-place
    script_dir = Path(__file__).parent
    input_dir = args.input_dir or str(script_dir / "QA" / "P2-cot")

    if args.output_dir:
        process_directory(input_dir, args.output_dir, in_place=False)
    else:
        # In-place cleaning
        print(f"Cleaning files IN-PLACE in: {input_dir}")
        print(f"{'='*60}")
        process_directory(input_dir, input_dir, in_place=True)


if __name__ == "__main__":
    main()
