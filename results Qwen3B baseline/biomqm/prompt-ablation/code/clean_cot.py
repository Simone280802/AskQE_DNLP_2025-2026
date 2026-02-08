"""
Script to extract final answers from CoT (Chain-of-Thought) responses.
It removes the reasoning chain and keeps only the text after "FINAL ANSWER:".

Usage:
    python clean_cot.py --input_dir "path/to/cot/results" --output_dir "path/to/clean/results"
"""

import json
import os
import re
import argparse
import glob

def extract_final_answer(text):
    """
    Extracts the answer from the CoT response.
    Expected format:
    Reasoning: ...
    FINAL ANSWER: ...
    """
    if not text:
        return ""
    
    # Pattern 1: Standard "FINAL ANSWER:"
    match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "Final Answer:" (case insensitive covered above)
    # Sometimes models might use **Final Answer:**
    match = re.search(r"\*\*FINAL ANSWER:\*\*\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: "Answer:"
    match = re.search(r"\nAnswer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: "Therefore, the answer is:"
    match = re.search(r"Therefore, the answer is:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 5: "The answer is:"
    match = re.search(r"The answer is:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 6: "Answer: ..." at the very start of line (if no prompt/reasoning)
    match = re.search(r"^Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: strict CoT might not have worked.
    # Heuristic: Take the last non-empty line? or just return the whole text?
    # For now, let's keep the whole text but mark it as "unparsed" in logs if needed.
    # Actually, for BioMQM, taking the last sentence often works best if explicit tag is missing.
    # But let's return the whole text for now to avoid losing data, 
    # unless it's very long (indicating reasoning is still there).
    return text.strip()

def process_file(input_file, output_file):
    print(f"Processing {input_file}...")
    
    extracted_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
                total_count += 1
                
                # Handle 'answers' field which can be a list or string
                raw_answers = data.get('answers', [])
                if isinstance(raw_answers, str):
                    raw_answers = [raw_answers]
                
                cleaned_answers = []
                for ans in raw_answers:
                    extracted = extract_final_answer(ans)
                    cleaned_answers.append(extracted)
                    
                    # Simple check if extraction happened (length reduced significantly)
                    if len(extracted) < len(ans) * 0.9: 
                        extracted_count += 1
                
                # Update data
                data['answers'] = cleaned_answers
                data['raw_answers'] = raw_answers # Keep original for debugging
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_file}")
                
    print(f"  Saved to {output_file}")
    print(f"  Extracted answers for {extracted_count}/{total_count} rows (heuristic check)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cleaned .jsonl files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all jsonl files
    files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    if not files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files to process.")
    
    for input_file in files:
        filename = os.path.basename(input_file)
        if "clean" in filename: # Avoid re-processing if running in same dir
            continue
            
        output_file = os.path.join(args.output_dir, f"clean-{filename}")
        process_file(input_file, output_file)

if __name__ == "__main__":
    main()
