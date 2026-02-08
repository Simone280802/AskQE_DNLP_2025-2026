"""
Analysis script for Prompt Ablation Study
Compares results across 3 strategies: fewshot, cot, concise
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Strategies to compare (baseline already done separately)
STRATEGIES = ["P1-fewshot", "P2-cot", "P3-concise"]
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def load_evaluation_results(base_path, strategy):
    """Load evaluation results for a strategy."""
    eval_path = os.path.join(base_path, strategy, "evaluation")
    
    results = {
        "strategy": strategy,
        "sbert": {},
        "string_comparison": {}
    }
    
    # Load SBERT summary
    sbert_file = os.path.join(eval_path, "sbert_summary_by_lang.csv")
    if os.path.exists(sbert_file):
        df = pd.read_csv(sbert_file)
        for _, row in df.iterrows():
            results["sbert"][row["lang"]] = row["avg_similarity"]
    
    # Load String Comparison summary
    sc_file = os.path.join(eval_path, "string_comparison_summary_by_lang.csv")
    if os.path.exists(sc_file):
        df = pd.read_csv(sc_file)
        for _, row in df.iterrows():
            results["string_comparison"][row["lang"]] = {
                "f1": row["avg_f1"],
                "em": row["avg_em"]
            }
    
    return results


def load_baseline_results(baseline_path):
    """Load baseline results for comparison."""
    results = {
        "strategy": "baseline",
        "sbert": {},
        "string_comparison": {}
    }
    
    sbert_file = os.path.join(baseline_path, "sbert_summary_by_lang.csv")
    if os.path.exists(sbert_file):
        df = pd.read_csv(sbert_file)
        for _, row in df.iterrows():
            results["sbert"][row["lang"]] = row["avg_similarity"]
    
    sc_file = os.path.join(baseline_path, "string_comparison_summary_by_lang.csv")
    if os.path.exists(sc_file):
        df = pd.read_csv(sc_file)
        for _, row in df.iterrows():
            results["string_comparison"][row["lang"]] = {
                "f1": row["avg_f1"],
                "em": row["avg_em"]
            }
    
    return results


def create_comparison_table(base_path, baseline_path, output_path):
    """Create comparison table across all strategies including baseline."""
    rows = []
    
    # Add baseline
    baseline = load_baseline_results(baseline_path)
    for lang in LANGUAGES:
        row = {
            "strategy": "baseline",
            "lang": lang,
            "sbert": baseline["sbert"].get(lang, None),
            "f1": baseline["string_comparison"].get(lang, {}).get("f1", None),
            "em": baseline["string_comparison"].get(lang, {}).get("em", None)
        }
        rows.append(row)
    
    # Add ablation strategies
    for strategy in STRATEGIES:
        results = load_evaluation_results(base_path, strategy)
        
        for lang in LANGUAGES:
            row = {
                "strategy": strategy,
                "lang": lang,
                "sbert": results["sbert"].get(lang, None),
                "f1": results["string_comparison"].get(lang, {}).get("f1", None),
                "em": results["string_comparison"].get(lang, {}).get("em", None)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved to: {output_path}")
    
    return df


def create_comparison_plots(df, output_dir):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_strategies = ["baseline"] + STRATEGIES
    
    # 1. SBERT Comparison Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index="strategy", columns="lang", values="sbert")
    pivot = pivot.reindex(all_strategies)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0.5, vmax=0.9)
    ax.set_title("SBERT Similarity by Strategy and Language")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_sbert_heatmap.png"), dpi=150)
    plt.close()
    
    # 2. F1 Comparison Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index="strategy", columns="lang", values="f1")
    pivot = pivot.reindex(all_strategies)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0.3, vmax=0.8)
    ax.set_title("F1 Score by Strategy and Language")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_f1_heatmap.png"), dpi=150)
    plt.close()
    
    # 3. Average Performance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_df = df.groupby("strategy")[["sbert", "f1", "em"]].mean().reindex(all_strategies)
    
    x = np.arange(len(all_strategies))
    width = 0.25
    
    ax.bar(x - width, avg_df["sbert"], width, label="SBERT", color="steelblue")
    ax.bar(x, avg_df["f1"], width, label="F1", color="forestgreen")
    ax.bar(x + width, avg_df["em"], width, label="EM", color="coral")
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_strategies, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Average Performance by Strategy")
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_avg_performance.png"), dpi=150)
    plt.close()
    
    # 4. Delta from Baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_avg = avg_df.loc["baseline"][["sbert", "f1", "em"]].values
    
    for i, strategy in enumerate(STRATEGIES):
        strategy_avg = avg_df.loc[strategy][["sbert", "f1", "em"]].values
        deltas = strategy_avg - baseline_avg
        
        x_pos = np.arange(3) + i * 0.25
        ax.bar(x_pos, deltas, 0.2, label=strategy)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks([0.25, 1.25, 2.25])
    ax.set_xticklabels(["ΔSBERT", "ΔF1", "ΔEM"])
    ax.set_ylabel("Delta from Baseline")
    ax.set_title("Improvement over Baseline")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_delta_from_baseline.png"), dpi=150)
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--base_path", type=str, required=True, 
                        help="Base path containing strategy folders")
    parser.add_argument("--baseline_path", type=str, required=True,
                        help="Path to baseline evaluation folder")
    parser.add_argument("--output_path", type=str, default="comparison_results.csv",
                        help="Output CSV path")
    parser.add_argument("--plot_dir", type=str, default="comparison_plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    df = create_comparison_table(args.base_path, args.baseline_path, args.output_path)
    create_comparison_plots(df, args.plot_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    all_strategies = ["baseline"] + STRATEGIES
    avg_df = df.groupby("strategy")[["sbert", "f1", "em"]].mean().reindex(all_strategies)
    print("\nAverage scores by strategy:")
    print(avg_df.round(3))
    
    # Find best strategy
    best_sbert = avg_df["sbert"].idxmax()
    best_f1 = avg_df["f1"].idxmax()
    best_em = avg_df["em"].idxmax()
    
    print(f"\nBest SBERT: {best_sbert} ({avg_df.loc[best_sbert, 'sbert']:.3f})")
    print(f"Best F1: {best_f1} ({avg_df.loc[best_f1, 'f1']:.3f})")
    print(f"Best EM: {best_em} ({avg_df.loc[best_em, 'em']:.3f})")


if __name__ == "__main__":
    main()
