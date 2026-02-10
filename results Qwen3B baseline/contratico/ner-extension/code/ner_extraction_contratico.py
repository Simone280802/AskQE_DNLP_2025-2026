"""
NER Extraction for contraTICO

Reads contraTICO JSONL files and extracts named entities from English source
sentences using d4data/biomedical-ner-all.

Output format is compatible with qg_entity_aware.py (uses 'src' field).

Usage:
    python ner_extraction_contratico.py \
        --input_path /path/to/contratico/en-es/alteration.jsonl \
        --output_path /path/to/ner_output.jsonl \
        [--sample_size 84] [--seed 42] [--max_samples N]
"""

import json
import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


MODEL_NAME = "d4data/biomedical-ner-all"


def load_ner_pipeline():
    """Load the NER pipeline."""
    print(f"Loading NER model: {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "ner",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        aggregation_strategy="simple",
        device=device
    )
    print(f"NER model loaded on {'GPU' if device == 0 else 'CPU'}")
    return ner


def aggregate_consecutive_entities(entities):
    """Aggregate consecutive entities of the same type."""
    if not entities:
        return []
    entities = sorted(entities, key=lambda x: x['start'])
    aggregated = []
    current = entities[0].copy()
    for next_entity in entities[1:]:
        if (next_entity['start'] <= current['end'] + 1 and
                next_entity['type'] == current['type']):
            current['text'] = current['text'] + next_entity['text']
            current['end'] = next_entity['end']
            current['score'] = round((current['score'] + next_entity['score']) / 2, 4)
        else:
            aggregated.append(current)
            current = next_entity.copy()
    aggregated.append(current)
    return aggregated


def extract_entities(ner_pipeline, text):
    """Extract entities from text."""
    if not text or not text.strip():
        return []
    try:
        results = ner_pipeline(text)
        entities = []
        for entity in results:
            entity_type = entity.get("entity_group", "UNKNOWN")
            if entity_type in ['O', '0', 'LABEL_0'] or entity_type.startswith('0'):
                continue
            score = entity.get("score", 0.0)
            if hasattr(score, 'item'):
                score = score.item()
            entity_text = entity.get("word", "").replace("##", "").strip()
            if not entity_text or len(entity_text) < 2:
                continue
            entities.append({
                "text": entity_text,
                "type": entity_type,
                "start": int(entity.get("start", 0)),
                "end": int(entity.get("end", 0)),
                "score": round(float(score), 4)
            })
        entities = aggregate_consecutive_entities(entities)
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="NER Extraction for contraTICO")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a contraTICO JSONL file (needs 'en' and 'id' fields)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file with entities")
    parser.add_argument("--sample_size", type=int, default=125,
                        help="Number of rows to sample (default: 84)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for testing")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return

    # Load data
    all_rows = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    print(f"Loaded {len(all_rows)} rows from {args.input_path}")

    # Sample rows (same indices as subset creation)
    n_sample = min(args.sample_size, len(all_rows))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(all_rows)), n_sample))
    sampled = [all_rows[i] for i in indices]

    # Deduplicate by 'en' field (should already be unique but just in case)
    seen = set()
    unique_rows = []
    for row in sampled:
        en = row.get('en', '')
        if en not in seen:
            seen.add(en)
            unique_rows.append(row)

    if args.max_samples:
        unique_rows = unique_rows[:args.max_samples]

    print(f"Processing {len(unique_rows)} unique source sentences")

    # Load NER model
    ner = load_ner_pipeline()

    # Extract entities
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    processed = 0
    total_entities = 0

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for row in unique_rows:
            processed += 1
            src = row.get('en', '')
            row_id = row.get('id', '')

            if processed % 20 == 0:
                print(f"[{processed}/{len(unique_rows)}] Processing...")

            entities = extract_entities(ner, src)
            total_entities += len(entities)

            output_row = {
                'id': row_id,
                'src': src,
                'entities': entities,
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 50}")
    print(f"NER Extraction Complete")
    print(f"Processed: {processed} sentences")
    print(f"Total entities found: {total_entities}")
    print(f"Output: {args.output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
