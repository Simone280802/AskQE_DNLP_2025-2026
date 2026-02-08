"""
EDA Visualizations for AskQE Baseline Results
Generates comprehensive plots for analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE, 'baseline', 'evaluation')
OUTPUT_PATH = os.path.join(BASE, 'eda_plots')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load data
print("Loading data...")
sbert = pd.read_csv(os.path.join(EVAL_PATH, 'sbert_all_languages.csv'))
sc = pd.read_csv(os.path.join(EVAL_PATH, 'string_comparison_all_languages.csv'))

# Convert EM to numeric if needed
sc['em'] = pd.to_numeric(sc['em'], errors='coerce').fillna(0)

print(f"SBERT samples: {len(sbert)}")
print(f"String Comparison samples: {len(sc)}")

# =============================================
# FIGURE 1: Distribution of SBERT Similarity
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a. Overall distribution
ax1 = axes[0]
ax1.hist(sbert['sbert_sim'], bins=50, edgecolor='white', alpha=0.7)
ax1.axvline(sbert['sbert_sim'].mean(), color='red', linestyle='--', label=f'Mean: {sbert["sbert_sim"].mean():.3f}')
ax1.axvline(sbert['sbert_sim'].median(), color='orange', linestyle='--', label=f'Median: {sbert["sbert_sim"].median():.3f}')
ax1.set_xlabel('SBERT Similarity')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of SBERT Similarity Scores')
ax1.legend()

# 1b. By language
ax2 = axes[1]
for lang in sbert['lang'].unique():
    subset = sbert[sbert['lang'] == lang]['sbert_sim']
    ax2.hist(subset, bins=30, alpha=0.5, label=lang)
ax2.set_xlabel('SBERT Similarity')
ax2.set_ylabel('Frequency')
ax2.set_title('SBERT Distribution by Language')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_sbert_distribution.png'), dpi=150)
plt.close()
print("Saved: 01_sbert_distribution.png")

# =============================================
# FIGURE 2: Boxplots by Language and Severity
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a. SBERT by Language
ax1 = axes[0, 0]
sbert.boxplot(column='sbert_sim', by='lang', ax=ax1)
ax1.set_xlabel('Language')
ax1.set_ylabel('SBERT Similarity')
ax1.set_title('SBERT Similarity by Language')
plt.suptitle('')

# 2b. SBERT by Severity
ax2 = axes[0, 1]
severity_order = ['Critical', 'Major', 'Minor', 'Neutral']
sbert['severity'] = pd.Categorical(sbert['severity'], categories=severity_order, ordered=True)
sbert.sort_values('severity').boxplot(column='sbert_sim', by='severity', ax=ax2)
ax2.set_xlabel('Severity')
ax2.set_ylabel('SBERT Similarity')
ax2.set_title('SBERT Similarity by Severity')
plt.suptitle('')

# 2c. F1 by Language
ax3 = axes[1, 0]
sc.boxplot(column='f1', by='lang', ax=ax3)
ax3.set_xlabel('Language')
ax3.set_ylabel('F1 Score')
ax3.set_title('F1 Score by Language')
plt.suptitle('')

# 2d. F1 by Severity
ax4 = axes[1, 1]
sc['severity'] = pd.Categorical(sc['severity'], categories=severity_order, ordered=True)
sc.sort_values('severity').boxplot(column='f1', by='severity', ax=ax4)
ax4.set_xlabel('Severity')
ax4.set_ylabel('F1 Score')
ax4.set_title('F1 Score by Severity')
plt.suptitle('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_boxplots.png'), dpi=150)
plt.close()
print("Saved: 02_boxplots.png")

# =============================================
# FIGURE 3: Heatmap Language x Severity
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3a. SBERT Heatmap
ax1 = axes[0]
pivot_sbert = sbert.pivot_table(values='sbert_sim', index='lang', columns='severity', aggfunc='mean')
pivot_sbert = pivot_sbert[['Critical', 'Major', 'Minor', 'Neutral']]
sns.heatmap(pivot_sbert, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, vmin=0.4, vmax=0.9)
ax1.set_title('SBERT Similarity: Language × Severity')

# 3b. F1 Heatmap
ax2 = axes[1]
pivot_f1 = sc.pivot_table(values='f1', index='lang', columns='severity', aggfunc='mean')
pivot_f1 = pivot_f1[['Critical', 'Major', 'Minor', 'Neutral']]
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, vmin=0.1, vmax=0.8)
ax2.set_title('F1 Score: Language × Severity')

# 3c. EM Heatmap
ax3 = axes[2]
pivot_em = sc.pivot_table(values='em', index='lang', columns='severity', aggfunc='mean')
pivot_em = pivot_em[['Critical', 'Major', 'Minor', 'Neutral']]
sns.heatmap(pivot_em, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, vmin=0.0, vmax=0.5)
ax3.set_title('Exact Match: Language × Severity')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_heatmaps.png'), dpi=150)
plt.close()
print("Saved: 03_heatmaps.png")

# =============================================
# FIGURE 4: Bar Chart Comparison by Language
# =============================================
fig, ax = plt.subplots(figsize=(12, 6))

langs = ['de', 'es', 'fr', 'ru', 'zh-CN']
x = np.arange(len(langs))
width = 0.25

sbert_means = [sbert[sbert['lang'] == l]['sbert_sim'].mean() for l in langs]
f1_means = [sc[sc['lang'] == l]['f1'].mean() for l in langs]
em_means = [sc[sc['lang'] == l]['em'].mean() for l in langs]

bars1 = ax.bar(x - width, sbert_means, width, label='SBERT Similarity', color='steelblue')
bars2 = ax.bar(x, f1_means, width, label='F1 Score', color='forestgreen')
bars3 = ax.bar(x + width, em_means, width, label='Exact Match', color='coral')

ax.set_xlabel('Language')
ax.set_ylabel('Score')
ax.set_title('Metrics Comparison by Language (Baseline)')
ax.set_xticks(x)
ax.set_xticklabels(langs)
ax.legend()
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_language_comparison.png'), dpi=150)
plt.close()
print("Saved: 04_language_comparison.png")

# =============================================
# FIGURE 5: Severity Impact Analysis
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

severities = ['Critical', 'Major', 'Minor', 'Neutral']
x = np.arange(len(severities))
width = 0.35

# 5a. SBERT and F1 by Severity
ax1 = axes[0]
sbert_sev = [sbert[sbert['severity'] == s]['sbert_sim'].mean() for s in severities]
f1_sev = [sc[sc['severity'] == s]['f1'].mean() for s in severities]

bars1 = ax1.bar(x - width/2, sbert_sev, width, label='SBERT', color='steelblue')
bars2 = ax1.bar(x + width/2, f1_sev, width, label='F1', color='forestgreen')
ax1.set_xlabel('Severity')
ax1.set_ylabel('Score')
ax1.set_title('Scores by Error Severity')
ax1.set_xticks(x)
ax1.set_xticklabels(severities)
ax1.legend()
ax1.set_ylim(0, 1)

# 5b. Sample counts by severity
ax2 = axes[1]
sev_counts = sbert['severity'].value_counts().reindex(severities)
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
bars = ax2.bar(severities, sev_counts.values, color=colors)
ax2.set_xlabel('Severity')
ax2.set_ylabel('Number of Samples')
ax2.set_title('Sample Distribution by Severity')

for bar, count in zip(bars, sev_counts.values):
    ax2.annotate(f'{count}',
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_severity_analysis.png'), dpi=150)
plt.close()
print("Saved: 05_severity_analysis.png")

# =============================================
# FIGURE 6: Correlation Analysis
# =============================================
fig, ax = plt.subplots(figsize=(8, 6))

# Merge SBERT and SC data for correlation
merged = sbert.merge(sc, on=['lang', 'severity'], how='inner', suffixes=('', '_sc'))
corr_data = merged[['sbert_sim', 'f1', 'em']].corr()

sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', ax=ax, 
            vmin=-1, vmax=1, center=0)
ax.set_title('Correlation Between Metrics')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_correlation.png'), dpi=150)
plt.close()
print("Saved: 06_correlation.png")

# =============================================
# FIGURE 7: Problem Areas Highlight
# =============================================
fig, ax = plt.subplots(figsize=(12, 6))

# Find problematic combinations (low scores)
pivot_f1 = sc.pivot_table(values='f1', index='lang', columns='severity', aggfunc='mean')
pivot_f1 = pivot_f1[['Critical', 'Major', 'Minor', 'Neutral']]

# Flatten for plotting
problems = []
for lang in pivot_f1.index:
    for sev in pivot_f1.columns:
        val = pivot_f1.loc[lang, sev]
        problems.append({'lang': lang, 'severity': sev, 'f1': val})

prob_df = pd.DataFrame(problems)
prob_df['combo'] = prob_df['lang'] + ' - ' + prob_df['severity']
prob_df = prob_df.sort_values('f1')

# Color by threshold
colors = ['red' if x < 0.4 else 'orange' if x < 0.6 else 'green' for x in prob_df['f1']]
bars = ax.barh(prob_df['combo'], prob_df['f1'], color=colors)
ax.axvline(0.4, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
ax.axvline(0.6, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
ax.set_xlabel('F1 Score')
ax.set_title('F1 Score by Language-Severity (Sorted)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_problem_areas.png'), dpi=150)
plt.close()
print("Saved: 07_problem_areas.png")

print("\n" + "="*50)
print(f"All plots saved to: {OUTPUT_PATH}")
print("="*50)
