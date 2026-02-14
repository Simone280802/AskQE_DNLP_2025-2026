"""
ContraTICO Baseline Mapping Script

Combines QG + QA source + QA BT results into a single mapped JSONL per config.

Usage:
    python mapping_contratico.py --base_dir /path/to/baseline
"""

import json
import os
import argparse
import glob


LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PERTURBATIONS = ["alteration", "omission"]
PIPELINES = ["vanilla", "atomic", "semantic"]


def load_qa_entries(filepath):
    """Load QA JSONL and return list of entries."""
    entries = []
    if not os.path.exists(filepath):
        return entries
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries


def main():
    parser = argparse.ArgumentParser(description="ContraTICO Baseline Mapping")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to baseline directory")
    args = parser.parse_args()

    base = args.base_dir
    mapping_dir = os.path.join(base, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)

    for pipeline in PIPELINES:
        # Load source QA
        source_file = os.path.join(base, "QA", "source", f"en-{pipeline}.jsonl")
        source_entries = load_qa_entries(source_file)

        if not source_entries:
            print(f"[{pipeline}] No source QA found at {source_file}, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Mapping pipeline: {pipeline}")
        print(f"Source entries: {len(source_entries)}")

        # Build source map by index
        source_map = {}
        for i, entry in enumerate(source_entries):
            row_idxs = entry.get("row_indexes", [i])
            for idx in row_idxs:
                source_map[idx] = entry

        # Process each language × perturbation
        for lang in LANGUAGES:
            for pert in PERTURBATIONS:
                bt_file = os.path.join(base, "QA", "bt", f"{lang}-{pipeline}-{pert}.jsonl")
                bt_entries = load_qa_entries(bt_file)

                if not bt_entries:
                    print(f"  [{lang}/{pert}] No BT QA found, skipping")
                    continue

                output_file = os.path.join(mapping_dir, f"{lang}-{pipeline}-{pert}.jsonl")
                rows_written = 0

                with open(output_file, 'w', encoding='utf-8') as f_out:
                    for j, bt_entry in enumerate(bt_entries):
                        row_idxs = bt_entry.get("row_indexes", [j])

                        for idx in row_idxs:
                            src_entry = source_map.get(idx, {})

                            output_row = {
                                "row_index": idx,
                                "src": src_entry.get("src", bt_entry.get("src", "")),
                                "questions": src_entry.get("questions", bt_entry.get("questions", [])),
                                "answers_src": src_entry.get("answers", []),
                                "answers_bt": bt_entry.get("answers", []),
                                "lang": lang,
                                "perturbation": pert,
                                "pipeline": pipeline
                            }

                            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
                            rows_written += 1

                print(f"  [{lang}/{pert}] {rows_written} rows -> {output_file}")

    print(f"\n{'='*50}")
    print("Mapping complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
