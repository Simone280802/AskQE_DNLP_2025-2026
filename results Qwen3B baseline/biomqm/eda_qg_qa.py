"""
EDA Visualizations for QG and QA Results
Analyzes question generation and answering quality
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
QG_PATH = os.path.join(BASE, 'baseline', 'QG', 'qwen-3b.jsonl')
QA_PATH = os.path.join(BASE, 'baseline', 'QA')
OUTPUT_PATH = os.path.join(BASE, 'eda_plots')
os.makedirs(OUTPUT_PATH, exist_ok=True)


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
        return [str(parsed)]
    except json.JSONDecodeError:
        return [field.strip()]


# =============================================
# LOAD QG DATA
# =============================================
print("Loading QG data...")
qg_data = []
with open(QG_PATH, encoding='utf-8') as f:
    for line in f:
        qg_data.append(json.loads(line))

print(f"QG samples: {len(qg_data)}")

# =============================================
# FIGURE 1: Questions per Sample Distribution
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parse questions correctly
q_counts = []
for item in qg_data:
    questions = parse_json_field(item.get('questions', ''))
    q_counts.append(len(questions))

ax1 = axes[0]
ax1.hist(q_counts, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
ax1.axvline(np.mean(q_counts), color='red', linestyle='--', label=f'Mean: {np.mean(q_counts):.1f}')
ax1.set_xlabel('Questions per Sample')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Questions per Sample')
ax1.legend()

# Stats box
stats_text = f'Min: {min(q_counts)}\nMax: {max(q_counts)}\nMean: {np.mean(q_counts):.1f}\nMedian: {np.median(q_counts):.1f}'
ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Boxplot
ax2 = axes[1]
ax2.boxplot(q_counts, vert=True)
ax2.set_ylabel('Questions per Sample')
ax2.set_title('Boxplot of Questions per Sample')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_qg_questions_distribution.png'), dpi=150)
plt.close()
print("Saved: 08_qg_questions_distribution.png")

# =============================================
# FIGURE 2: Question Length Distribution
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

all_q_lengths = []
for item in qg_data:
    questions = parse_json_field(item.get('questions', ''))
    for q in questions:
        all_q_lengths.append(len(str(q)))

ax1 = axes[0]
ax1.hist(all_q_lengths, bins=50, edgecolor='white', alpha=0.7, color='forestgreen')
ax1.axvline(np.mean(all_q_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(all_q_lengths):.1f}')
ax1.set_xlabel('Question Length (characters)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Question Lengths')
ax1.legend()

# Sample questions
ax2 = axes[1]
sample_qs = []
for item in qg_data[:10]:
    questions = parse_json_field(item.get('questions', ''))
    if questions:
        sample_qs.append(questions[0][:50] + '...' if len(questions[0]) > 50 else questions[0])
    if len(sample_qs) >= 5:
        break

ax2.axis('off')
ax2.set_title('Sample Questions')
text = '\n\n'.join([f'{i+1}. {q}' for i, q in enumerate(sample_qs)])
ax2.text(0.1, 0.9, text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', wrap=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08b_question_lengths.png'), dpi=150)
plt.close()
print("Saved: 08b_question_lengths.png")

# =============================================
# LOAD QA DATA
# =============================================
print("\nLoading QA data...")
qa_files = ['source-vanilla.jsonl', 'bt-de-vanilla.jsonl', 'bt-es-vanilla.jsonl', 
            'bt-fr-vanilla.jsonl', 'bt-ru-vanilla.jsonl', 'bt-zh-CN-vanilla.jsonl']

qa_stats = {}
for qa_file in qa_files:
    fpath = os.path.join(QA_PATH, qa_file)
    if not os.path.exists(fpath):
        continue
    
    with open(fpath, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    total_answers = 0
    empty_answers = 0
    answer_lengths = []
    
    for item in data:
        answers = parse_json_field(item.get('answers', ''))
        
        for ans in answers:
            total_answers += 1
            ans_str = str(ans).strip()
            answer_lengths.append(len(ans_str))
            if not ans_str or ans_str == '':
                empty_answers += 1
    
    lang = qa_file.replace('bt-', '').replace('-vanilla.jsonl', '')
    if lang == 'source':
        lang = 'source'
    
    qa_stats[lang] = {
        'samples': len(data),
        'total_answers': total_answers,
        'empty_pct': 100 * empty_answers / max(total_answers, 1),
        'avg_ans_len': np.mean(answer_lengths) if answer_lengths else 0
    }

print(f"Loaded {len(qa_stats)} QA files")

# =============================================
# FIGURE 3: QA Statistics by Language
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

langs = list(qa_stats.keys())
x = np.arange(len(langs))

# 3a. Total Answers
ax1 = axes[0, 0]
values = [qa_stats[l]['total_answers'] for l in langs]
bars = ax1.bar(x, values, color='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(langs)
ax1.set_ylabel('Count')
ax1.set_title('Total Answers per Language')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,}',
             ha='center', va='bottom', fontsize=8)

# 3b. Empty Answers %
ax2 = axes[0, 1]
values = [qa_stats[l]['empty_pct'] for l in langs]
colors = ['red' if v > 15 else 'orange' if v > 10 else 'green' for v in values]
bars = ax2.bar(x, values, color=colors)
ax2.set_xticks(x)
ax2.set_xticklabels(langs)
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Empty Answers Rate')
ax2.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Warning: 10%')
ax2.legend()
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}%',
             ha='center', va='bottom', fontsize=9)

# 3c. Average Answer Length
ax3 = axes[1, 0]
values = [qa_stats[l]['avg_ans_len'] for l in langs]
bars = ax3.bar(x, values, color='forestgreen')
ax3.set_xticks(x)
ax3.set_xticklabels(langs)
ax3.set_ylabel('Characters')
ax3.set_title('Average Answer Length')
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
             ha='center', va='bottom', fontsize=9)

# 3d. Samples per Language
ax4 = axes[1, 1]
values = [qa_stats[l]['samples'] for l in langs]
bars = ax4.bar(x, values, color='coral')
ax4.set_xticks(x)
ax4.set_xticklabels(langs)
ax4.set_ylabel('Count')
ax4.set_title('Samples per Language')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '09_qa_statistics.png'), dpi=150)
plt.close()
print("Saved: 09_qa_statistics.png")

# =============================================
# FIGURE 4: Answer Length Distribution
# =============================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get all answer lengths for source
source_ans_lens = []
fpath = os.path.join(QA_PATH, 'source-vanilla.jsonl')
if os.path.exists(fpath):
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            answers = parse_json_field(item.get('answers', ''))
            for ans in answers:
                source_ans_lens.append(len(str(ans)))

ax.hist(source_ans_lens, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
ax.axvline(np.mean(source_ans_lens), color='red', linestyle='--', label=f'Mean: {np.mean(source_ans_lens):.1f}')
ax.axvline(np.median(source_ans_lens), color='orange', linestyle='--', label=f'Median: {np.median(source_ans_lens):.1f}')
ax.set_xlabel('Answer Length (characters)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Answer Lengths (Source)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '10_answer_lengths.png'), dpi=150)
plt.close()
print("Saved: 10_answer_lengths.png")

# =============================================
# Print Summary
# =============================================
print("\n" + "="*60)
print("QG/QA EDA SUMMARY (CORRECTED)")
print("="*60)
print(f"\nQG Analysis:")
print(f"  - Total samples: {len(qg_data)}")
print(f"  - Questions per sample: min={min(q_counts)}, max={max(q_counts)}, avg={np.mean(q_counts):.1f}")
print(f"  - Total questions generated: {sum(q_counts)}")
print(f"  - Avg question length: {np.mean(all_q_lengths):.1f} chars")

print(f"\nQA Analysis:")
for lang, stats in qa_stats.items():
    print(f"  {lang}: {stats['samples']} samples, {stats['total_answers']} answers, {stats['empty_pct']:.1f}% empty, avg len={stats['avg_ans_len']:.1f}")

print(f"\nPlots saved to: {OUTPUT_PATH}")
print("="*60)
