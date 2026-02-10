"""
Create a contraTICO subset (~5,040 rows) for extension experiments.
Samples 42 rows per perturbation file, maintaining alignment across
QA source, QA bt, and QG files.

Output: results Qwen3B baseline/contratico/{QA,QG}/
"""

import json
import os
import random

# ── Configuration ──────────────────────────────────────────────────
SEED = 42
ROWS_PER_PERTURBATION = 84
LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
CONFIGS = ["vanilla", "atomic", "semantic"]
PERTURBATIONS = [
    "alteration", "expansion_impact", "expansion_noimpact",
    "intensifier", "omission", "spelling", "synonym", "word_order",
]
TOTAL_ROWS = 971  # rows per perturbation file in the original dataset

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(SCRIPT_DIR)  # results Qwen3B baseline

BASELINE_QA_SOURCE = os.path.join(RESULTS_DIR, "QA", "source")
BASELINE_QA_BT = os.path.join(RESULTS_DIR, "QA", "bt")
BASELINE_QG = os.path.join(RESULTS_DIR, "QG")

OUTPUT_DIR = SCRIPT_DIR  # results Qwen3B baseline/contratico
OUTPUT_QA_SOURCE = os.path.join(OUTPUT_DIR, "QA", "source")
OUTPUT_QA_BT = os.path.join(OUTPUT_DIR, "QA", "bt")
OUTPUT_QG = os.path.join(OUTPUT_DIR, "QG")


def read_jsonl(filepath):
    """Read all lines from a JSONL file."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, filepath):
    """Write rows to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def sample_indices(seed, n_total, n_sample):
    """Generate a sorted list of sampled indices."""
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_total), n_sample))
    return indices


def select_rows(rows, indices):
    """Select rows at the given indices."""
    return [rows[i] for i in indices]


def main():
    print("=" * 60)
    print("contraTICO Subset Creator")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Rows per perturbation: {ROWS_PER_PERTURBATION}")
    print(f"Languages: {LANGUAGES}")
    print(f"Configs: {CONFIGS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Use the same sampled indices for all files (they all have 971 rows
    # aligned by index — same source sentence at each index)
    indices = sample_indices(SEED, TOTAL_ROWS, ROWS_PER_PERTURBATION)
    print(f"Sampled {len(indices)} indices (first 5: {indices[:5]}...)")
    print()

    total_files = 0
    total_rows = 0

    # ── 1) QG files ────────────────────────────────────────────────
    print("─── QG ───")
    for config in CONFIGS:
        src_file = os.path.join(BASELINE_QG, f"{config}_qwen-3b.jsonl")
        dst_file = os.path.join(OUTPUT_QG, f"{config}_qwen-3b.jsonl")

        rows = read_jsonl(src_file)
        sampled = select_rows(rows, indices)
        write_jsonl(sampled, dst_file)
        print(f"  {config}: {len(sampled)} rows")
        total_files += 1
        total_rows += len(sampled)

    # ── 2) QA source files ─────────────────────────────────────────
    print("─── QA source ───")
    for config in CONFIGS:
        src_file = os.path.join(BASELINE_QA_SOURCE, f"en-{config}.jsonl")
        dst_file = os.path.join(OUTPUT_QA_SOURCE, f"en-{config}.jsonl")

        rows = read_jsonl(src_file)
        sampled = select_rows(rows, indices)
        write_jsonl(sampled, dst_file)
        print(f"  en-{config}: {len(sampled)} rows")
        total_files += 1
        total_rows += len(sampled)

    # ── 3) QA bt files ─────────────────────────────────────────────
    print("─── QA bt ───")
    for lang in LANGUAGES:
        for config in CONFIGS:
            for pert in PERTURBATIONS:
                filename = f"{lang}-{config}-{pert}.jsonl"
                src_file = os.path.join(BASELINE_QA_BT, lang, config, filename)
                dst_file = os.path.join(OUTPUT_QA_BT, lang, config, filename)

                if not os.path.exists(src_file):
                    print(f"  MISSING: {src_file}")
                    continue

                rows = read_jsonl(src_file)
                sampled = select_rows(rows, indices)
                write_jsonl(sampled, dst_file)
                total_files += 1
                total_rows += len(sampled)

            print(f"  {lang}/{config}: {ROWS_PER_PERTURBATION * len(PERTURBATIONS)} rows ({len(PERTURBATIONS)} files)")

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Done! Created {total_files} files")
    print(f"Total QA bt rows: {total_rows - (len(CONFIGS) * ROWS_PER_PERTURBATION * 2)} rows")
    print(f"Total rows (incl. QG + QA source): {total_rows}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
