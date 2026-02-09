"""
Prompt Ablation Study - Mapping & Evaluation Script
This script runs mapping and evaluation for ALL 3 prompt strategies.

Usage:
  python run_mapping_evaluation.py [--project_root /path/to/project]

Prerequisites: Run notebooks 1-6 first (all source and BT QA).
"""

import os
import sys
import json
import subprocess
import shutil

# ========================================
# ENVIRONMENT SETUP
# ========================================

IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle')

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'Local'}")

if IN_KAGGLE:
    PROJECT_ROOT = '/kaggle/working/askqe'
    if not os.path.exists(PROJECT_ROOT):
        subprocess.run(['git', 'clone', 'https://github.com/Simone280802/AskQE_DNLP_2025-2026.git', PROJECT_ROOT], check=True)
elif IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/askqe'
    if not os.path.exists(PROJECT_ROOT):
        subprocess.run(['git', 'clone', 'https://github.com/Simone280802/AskQE_DNLP_2025-2026.git', PROJECT_ROOT], check=True)
else:
    # Local: assume script is in prompt-ablation folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

print(f'Project root: {PROJECT_ROOT}')

# ========================================
# PATH CONFIGURATION
# ========================================

ABLATION_DIR = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'biomqm', 'prompt-ablation')
BASELINE_EVAL = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'biomqm', 'baseline', 'evaluation')
QG_PATH = os.path.join(PROJECT_ROOT, 'results Qwen3B baseline', 'biomqm', 'baseline', 'QG', 'qwen-3b.jsonl')
ORIGINAL_DATASET = os.path.join(PROJECT_ROOT, 'biomqm', 'dev_with_backtranslation.jsonl')

STRATEGIES = ['P1-fewshot', 'P2-cot', 'P3-concise']
LANGUAGES = ['de', 'es', 'fr', 'ru', 'zh-CN']

# Add prompt-ablation dir to path for imports
if ABLATION_DIR not in sys.path:
    sys.path.insert(0, ABLATION_DIR)

print(f'ABLATION_DIR: {ABLATION_DIR}')
print(f'BASELINE_EVAL: {BASELINE_EVAL}')
print(f'ORIGINAL_DATASET: {ORIGINAL_DATASET}')

# ========================================
# VERIFY QA FILES EXIST
# ========================================

print('\n=== Checking QA files ===')
all_ok = True
for strategy in STRATEGIES:
    print(f'\n{strategy}:')
    # Source
    src_file = os.path.join(ABLATION_DIR, 'QA', strategy, f'source-{strategy}.jsonl')
    exists = os.path.exists(src_file)
    print(f"  {'✓' if exists else '✗'} source")
    if not exists: all_ok = False
    # BT
    for lang in LANGUAGES:
        bt_file = os.path.join(ABLATION_DIR, 'QA', strategy, f'bt-{lang}-{strategy}.jsonl')
        exists = os.path.exists(bt_file)
        print(f"  {'✓' if exists else '✗'} bt-{lang}")
        if not exists: all_ok = False
if all_ok:
    print('\n=== All QA files found! ===')
else:
    print('\n=== WARNING: Some files missing! ===')

# ========================================
# RUN MAPPING
# ========================================

from mapping import run_mapping

print('\n' + '='*60)
print('RUNNING MAPPING FOR ALL STRATEGIES')
print('='*60)

for strategy in STRATEGIES:
    print(f'\n=== Mapping {strategy} ===')
    
    # Paths for this strategy
    qa_dir = os.path.join(ABLATION_DIR, 'QA', strategy)
    output_dir = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped')
    
    # Run mapping using the external module
    rows = run_mapping(
        strategy=strategy,
        qg_path=QG_PATH,
        original_dataset_path=ORIGINAL_DATASET,
        qa_dir=qa_dir,
        output_dir=output_dir
    )
    print(f'Completed {strategy}: {rows} rows mapped')

# ========================================
# RUN SBERT EVALUATION
# ========================================

print('\n' + '='*60)
print('RUNNING SBERT EVALUATION')
print('='*60)

for strategy in STRATEGIES:
    print(f'\n=== SBERT {strategy} ===')
    mapping_file = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', f'all-{strategy}.jsonl')
    output_dir = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', 'evaluation', 'sbert')
    
    # Use the custom script
    sbert_script = os.path.join(ABLATION_DIR, 'evaluation', 'sbert.py')
    
    cmd = [
        sys.executable, '-u',
        sbert_script,
        '--input_path', mapping_file,
        '--output_dir', output_dir
    ]
    subprocess.run(cmd, check=True)
    print(f'Done {strategy}!')

# ========================================
# RUN STRING COMPARISON EVALUATION
# ========================================

print('\n' + '='*60)
print('RUNNING STRING COMPARISON EVALUATION')
print('='*60)

for strategy in STRATEGIES:
    print(f'\n=== String Comparison {strategy} ===')
    mapping_file = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', f'all-{strategy}.jsonl')
    output_dir = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', 'evaluation', 'string-comparison')
    
    # Use the custom script
    sc_script = os.path.join(ABLATION_DIR, 'evaluation', 'string_comparison.py')
    
    cmd = [
        sys.executable, '-u',
        sc_script,
        '--input_path', mapping_file,
        '--output_dir', output_dir
    ]
    subprocess.run(cmd, check=True)
    print(f'Done {strategy}!')

# ========================================
# RESULTS COMPARISON
# ========================================

import pandas as pd

print('\n' + '='*60)
print('ABLATION STUDY RESULTS')
print('='*60)

results = []

# Load baseline
baseline_sbert = os.path.join(BASELINE_EVAL, 'sbert_summary_by_lang.csv')
baseline_sc = os.path.join(BASELINE_EVAL, 'string_comparison_summary_by_lang.csv')

if os.path.exists(baseline_sbert):
    sbert_df = pd.read_csv(baseline_sbert)
    avg_sbert = sbert_df['avg_similarity'].mean()
else:
    avg_sbert = None

if os.path.exists(baseline_sc):
    sc_df = pd.read_csv(baseline_sc)
    avg_f1 = sc_df['avg_f1'].mean()
    avg_em = sc_df['avg_em'].mean()
    avg_bleu = sc_df['avg_bleu'].mean() if 'avg_bleu' in sc_df.columns else None
    avg_chrf = sc_df['avg_chrf'].mean() if 'avg_chrf' in sc_df.columns else None
else:
    avg_f1, avg_em, avg_bleu, avg_chrf = None, None, None, None

results.append({'strategy': 'baseline', 'sbert': avg_sbert, 'f1': avg_f1, 'em': avg_em, 'bleu': avg_bleu, 'chrf': avg_chrf})

# Load ablation results
for strategy in STRATEGIES:
    sbert_file = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', 'evaluation', 'sbert_summary_by_lang.csv')
    sc_file = os.path.join(ABLATION_DIR, 'QA', strategy, 'mapped', 'evaluation', 'string_comparison_summary_by_lang.csv')
    
    if os.path.exists(sbert_file):
        sbert_df = pd.read_csv(sbert_file)
        avg_sbert = sbert_df['avg_similarity'].mean()
    else:
        avg_sbert = None
    
    if os.path.exists(sc_file):
        sc_df = pd.read_csv(sc_file)
        avg_f1 = sc_df['avg_f1'].mean()
        avg_em = sc_df['avg_em'].mean()
        avg_bleu = sc_df['avg_bleu'].mean() if 'avg_bleu' in sc_df.columns else None
        avg_chrf = sc_df['avg_chrf'].mean() if 'avg_chrf' in sc_df.columns else None
    else:
        avg_f1, avg_em, avg_bleu, avg_chrf = None, None, None, None
    
    results.append({'strategy': strategy, 'sbert': avg_sbert, 'f1': avg_f1, 'em': avg_em, 'bleu': avg_bleu, 'chrf': avg_chrf})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ========================================
# SUMMARY
# ========================================

print('\n' + '='*60)
print('ABLATION STUDY COMPLETE')
print('='*60)
print(f'\nResults saved to: {ABLATION_DIR}')
print('\nFor each strategy (P1-fewshot, P2-cot, P3-concise):')
print('  - QA/{strategy}/mapped/all-{strategy}.jsonl')
print('  - QA/{strategy}/mapped/evaluation/sbert/')
print('  - QA/{strategy}/mapped/evaluation/string-comparison/')
