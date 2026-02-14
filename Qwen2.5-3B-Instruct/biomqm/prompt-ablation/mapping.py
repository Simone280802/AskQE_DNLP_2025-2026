"""
BIOMQM QA Mapping Script (Index-based) for Prompt Ablation
Combines source and bt answers to reconstruct the full dataset using row indexes.

Usage (standalone):
  python mapping.py --strategy P1-fewshot --project_root /path/to/project

Usage (as module):
  from mapping import run_mapping
  run_mapping(strategy, qg_path, original_dataset_path, qa_dir, output_dir)
"""

import json
import os
import argparse

LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]
SEVERITY_ORDER = {"Critical": 4, "Major": 3, "Minor": 2, "Neutral": 1}


def load_qa_map(file_path):
    """
    Loads QA file and builds a map: row_index -> {answers, questions}
    """
    qa_map = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return qa_map
        
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                answers = data.get('answers', [])
                questions = data.get('questions', [])
                row_indexes = data.get('row_indexes', [])
                
                for idx in row_indexes:
                    qa_map[idx] = {
                        'answers': answers,
                        'questions': questions
                    }
            except json.JSONDecodeError:
                continue
    return qa_map


def run_mapping(strategy, qg_path, original_dataset_path, qa_dir, output_dir):
    """
    Run the mapping process for a given strategy.
    
    Args:
        strategy: Strategy name (e.g., 'P1-fewshot', 'P2-cot', 'P3-concise')
        qg_path: Path to QG file (e.g., 'baseline/QG/qwen-3b.jsonl')
        original_dataset_path: Path to original dataset (e.g., 'biomqm/dev_with_backtranslation.jsonl')
        qa_dir: Directory containing QA results for this strategy
        output_dir: Directory to save the mapped output
        
    Returns:
        Number of rows written
    """
    print(f"\n{'='*50}")
    print(f"MAPPING - Strategy: {strategy}")
    print(f"{'='*50}")
    
    # Validate inputs
    if not os.path.exists(original_dataset_path):
        print(f"Original dataset file not found at: {original_dataset_path}")
        return 0
    
    # 1. Load Source QA Map
    source_file = os.path.join(qa_dir, f"source-{strategy}.jsonl")
    if not os.path.exists(source_file):
        print(f"Source answers file not found: {source_file}")
        return 0
        
    source_map = load_qa_map(source_file)
    print(f"Source Map size: {len(source_map)}")
    
    # 2. Load BT QA Map (all languages)
    bt_map = {}
    for lang in LANGUAGES:
        bt_file = os.path.join(qa_dir, f"bt-{lang}-{strategy}.jsonl")
        lang_map = load_qa_map(bt_file)
        bt_map.update(lang_map)
    print(f"BT Map size: {len(bt_map)}")

    # 3. Process Original File
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all-{strategy}.jsonl")
    
    print(f"Processing original file: {original_dataset_path}")
    
    rows_written = 0
    missing_source = 0
    missing_bt = 0
    
    with open(original_dataset_path, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            try:
                original_row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Lookup answers by Index
            src_qa_data = source_map.get(idx, {})
            bt_qa_data = bt_map.get(idx, {})
            
            answer_src = src_qa_data.get('answers', [])
            answer_bt = bt_qa_data.get('answers', [])
            
            # Questions: Prefer Source, then BT, then original (if exists)
            questions = src_qa_data.get('questions') or bt_qa_data.get('questions') or original_row.get('questions', [])
            
            if not answer_src: missing_source += 1
            if not answer_bt: missing_bt += 1
            
            # Build Output Row
            errors = original_row.get('errors_tgt', [])
            all_severities = [e.get('severity', 'Neutral') for e in errors]
            if not all_severities:
                all_severities = ["Neutral"]

            output_row = {
                'src': original_row.get('src', ''),
                'bt_tgt': original_row.get('bt_tgt', ''),
                'lang_tgt': original_row.get('lang_tgt', ''),
                'questions': questions,
                'answers_src': answer_src,
                'answers_bt': answer_bt,
                'severities': all_severities,
                'docID': original_row.get('doc_id', ''),
                'system': original_row.get('system', ''),
                'strategy': strategy
            }
            
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
            rows_written += 1
            
    print(f"\n{'='*50}")
    print(f"MAPPING COMPLETED (Index-based)")
    print(f"Output: {output_file}")
    print(f"Total rows written: {rows_written}")
    if missing_source: print(f"WARNING: {missing_source} rows missing Source answers")
    if missing_bt: print(f"WARNING: {missing_bt} rows missing BT answers")
    print(f"{'='*50}")
    
    return rows_written


def main():
    parser = argparse.ArgumentParser(description="BIOMQM QA Mapping Script (Index-based) for Prompt Ablation")
    parser.add_argument("--strategy", type=str, required=True,
                        help="Strategy name (P1-fewshot, P2-cot, P3-concise)")
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory")
    parser.add_argument("--qg_path", type=str, default=None,
                        help="Path to QG file (overrides default)")
    parser.add_argument("--original_dataset", type=str, default=None,
                        help="Path to original dataset (overrides default)")
    parser.add_argument("--qa_dir", type=str, default=None,
                        help="Path to QA directory for strategy (overrides default)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for mapped file (overrides default)")
    args = parser.parse_args()
    
    # Setup paths
    if args.project_root:
        project_root = args.project_root
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up from prompt-ablation to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    
    # Default paths based on project structure
    qg_path = args.qg_path or os.path.join(
        project_root, "results Qwen3B baseline", "biomqm", "baseline", "QG", "qwen-3b.jsonl"
    )
    original_dataset = args.original_dataset or os.path.join(
        project_root, "biomqm", "dev_with_backtranslation.jsonl"
    )
    qa_dir = args.qa_dir or os.path.join(
        project_root, "results Qwen3B baseline", "biomqm", "prompt-ablation", "QA", args.strategy
    )
    output_dir = args.output_dir or os.path.join(
        project_root, "results Qwen3B baseline", "biomqm", "prompt-ablation", "QA", args.strategy, "mapped"
    )
    
    run_mapping(args.strategy, qg_path, original_dataset, qa_dir, output_dir)


if __name__ == "__main__":
    main()
