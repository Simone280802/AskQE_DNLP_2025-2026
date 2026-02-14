"""
QA Script for Prompt Ablation Study (contraTICO)
Generates answers using different prompt strategies.

For SOURCE mode: reads QG file, samples 42 rows (seed 42), answers questions
                 about the English source sentence using the alternate prompt.

For BT mode: reads the original contraTICO perturbation files to get the
             bt sentences (pert_{lang}), then answers the same questions
             about the perturbed sentence using the alternate prompt.

Usage:
  python qa_ablation_contratico.py --strategy P1-fewshot --mode source \
      --config vanilla --baseline_dir /path/to/baseline \
      --contratico_dir /path/to/contratico \
      --output_dir /path/to/prompt-ablation --max_rows 42 --seed 42

  python qa_ablation_contratico.py --strategy P1-fewshot --mode bt \
      --config vanilla --lang es --baseline_dir /path/to/baseline \
      --contratico_dir /path/to/contratico \
      --output_dir /path/to/prompt-ablation --max_rows 42 --seed 42
"""

import torch
import json
import os
import argparse
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import get_prompt, PROMPTS

model_id = "Qwen/Qwen2.5-3B-Instruct"
LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
CONFIGS = ["vanilla", "atomic", "semantic"]
PERTURBATIONS = [
    "alteration", "expansion_impact", "expansion_noimpact",
    "intensifier", "omission", "spelling", "synonym", "word_order",
]


def parse_questions(questions_field):
    """Parse questions from string or list."""
    if isinstance(questions_field, list):
        return questions_field
    if not questions_field or str(questions_field).strip() == "":
        return []
    try:
        questions = json.loads(questions_field)
        if isinstance(questions, list):
            return questions
        return [str(questions)]
    except (json.JSONDecodeError, TypeError):
        return [str(questions_field).strip()]


def generate_single_answer(tokenizer, model, device, sentence, question, strategy):
    """Generate answer for a single question using specified strategy."""
    prompt_template = get_prompt(strategy)
    prompt = prompt_template.format(sentence=sentence, question=question)

    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer questions directly and concisely."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)

    if answer:
        answer = answer.strip().strip('"\'')

    return answer


def generate_answers_for_questions(tokenizer, model, device, sentence, questions, strategy):
    """Generate answers for all questions."""
    parsed = parse_questions(questions)
    if not parsed:
        return []

    answers = []
    for q in parsed:
        answer = generate_single_answer(tokenizer, model, device, sentence, q, strategy)
        answers.append(answer)

    return answers


def get_sampled_ids(qg_path, max_rows, seed):
    """Load QG file and return sampled IDs + row data."""
    all_rows = []
    with open(qg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    if len(all_rows) > max_rows:
        rng = random.Random(seed)
        sampled = rng.sample(all_rows, max_rows)
    else:
        sampled = all_rows

    return {row['id'] for row in sampled}, {row['id']: row for row in sampled}


def process_source_qa(tokenizer, model, device, baseline_dir, output_dir, config, strategy, max_rows, seed):
    """Process source QA with specified strategy for a given config."""

    qg_path = os.path.join(baseline_dir, "QG", f"{config}_qwen-3b.jsonl")
    if not os.path.exists(qg_path):
        print(f"ERROR: QG file not found: {qg_path}")
        return

    # Sample rows (same seed as metrics-extension)
    sampled_ids, sampled_rows = get_sampled_ids(qg_path, max_rows, seed)
    print(f"Sampled {len(sampled_ids)} rows from {qg_path} (seed={seed})")

    # Output path
    output_file = os.path.join(output_dir, "QA", strategy, "source", f"en-{config}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check for resume
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data.get('id', ''))
        print(f"Resuming: {len(processed_ids)} already processed")

    remaining = [rid for rid in sampled_ids if rid not in processed_ids]
    print(f"To process: {len(remaining)} rows")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, row_id in enumerate(remaining):
            row = sampled_rows[row_id]
            questions = parse_questions(row.get('questions', []))
            sentence = row.get('en', '')

            print(f"[{i+1}/{len(remaining)}] {strategy}/{config} source - {len(questions)} questions...")

            answers = generate_answers_for_questions(
                tokenizer, model, device, sentence, questions, strategy
            )

            if answers:
                print(f"> First answer: {answers[0][:60]}...")

            output_row = {
                'id': row_id,
                'en': sentence,
                'questions': row.get('questions', []),
                'answers': answers,
                'strategy': strategy
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')

    print(f"\nSource QA ({strategy}/{config}) completed. Output: {output_file}")


def process_bt_qa(tokenizer, model, device, baseline_dir, contratico_dir, output_dir,
                  config, lang, strategy, max_rows, seed):
    """Process BT QA for a specific language with specified strategy.

    Reads bt sentences from the original contraTICO perturbation files
    (contratico/en-{lang}/{pert}.jsonl) which contain the 'pert_{lang}' field.
    """

    # Get sampled IDs (same as source)
    qg_path = os.path.join(baseline_dir, "QG", f"{config}_qwen-3b.jsonl")
    if not os.path.exists(qg_path):
        print(f"ERROR: QG file not found: {qg_path}")
        return

    sampled_ids, sampled_rows = get_sampled_ids(qg_path, max_rows, seed)
    print(f"Using {len(sampled_ids)} sampled IDs for {lang}/{config} (seed={seed})")

    pert_field = f"pert_{lang}"

    for pert in PERTURBATIONS:
        # Read original contraTICO file to get bt sentences
        contratico_file = os.path.join(contratico_dir, f"en-{lang}", f"{pert}.jsonl")
        if not os.path.exists(contratico_file):
            print(f"SKIP: {contratico_file} not found")
            continue

        # Load contraTICO data indexed by id
        contratico_by_id = {}
        with open(contratico_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    contratico_by_id[row.get('id', '')] = row

        # Output path
        bt_filename = f"{lang}-{config}-{pert}.jsonl"
        output_file = os.path.join(
            output_dir, "QA", strategy, "bt", lang, config, bt_filename
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check for resume
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    processed_ids.add(data.get('id', ''))

        # Filter to sampled IDs that exist in contraTICO and not yet processed
        to_process = []
        for row_id in sampled_ids:
            if row_id in processed_ids:
                continue
            if row_id not in contratico_by_id:
                continue
            to_process.append(row_id)

        if not to_process:
            n_matched = len([rid for rid in sampled_ids if rid in contratico_by_id])
            print(f"[{lang}/{config}/{pert}] all {n_matched} rows done")
            continue

        print(f"[{lang}/{config}/{pert}] Processing {len(to_process)} rows...")

        with open(output_file, 'a', encoding='utf-8') as f_out:
            for j, row_id in enumerate(to_process):
                ctic_row = contratico_by_id[row_id]
                qg_row = sampled_rows[row_id]

                # bt sentence = perturbed translation
                bt_sentence = ctic_row.get(pert_field, '')
                questions = parse_questions(qg_row.get('questions', []))

                if j < 2:
                    print(f"  [{j+1}/{len(to_process)}] {strategy}/{lang}/{config}/{pert} - {len(questions)} q")
                    if bt_sentence:
                        print(f"  > BT sentence: {bt_sentence[:60]}...")

                answers = generate_answers_for_questions(
                    tokenizer, model, device, bt_sentence, questions, strategy
                )

                output_row = {
                    'id': row_id,
                    'en': qg_row.get('en', ''),
                    'questions': qg_row.get('questions', []),
                    'answers': answers,
                    'strategy': strategy
                }
                f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')

        print(f"  Saved: {output_file}")

    print(f"\nBT QA ({strategy}/{lang}/{config}) completed.")


def main():
    parser = argparse.ArgumentParser(description="QA Script for Prompt Ablation - contraTICO")
    parser.add_argument("--strategy", type=str, required=True, choices=list(PROMPTS.keys()),
                        help=f"Prompt strategy: {list(PROMPTS.keys())}")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' or 'bt'")
    parser.add_argument("--config", type=str, required=True, choices=CONFIGS,
                        help="QG config: vanilla, atomic, semantic")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for bt mode")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Path to contraTICO baseline directory")
    parser.add_argument("--contratico_dir", type=str, default=None,
                        help="Path to original contraTICO data (en-{lang}/ dirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prompt-ablation")
    parser.add_argument("--max_rows", type=int, default=42,
                        help="Max rows to sample (default: 42)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required when --mode is 'bt'")
    if args.mode == "bt" and not args.contratico_dir:
        parser.error("--contratico_dir is required when --mode is 'bt'")

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Strategy: {args.strategy} - {PROMPTS[args.strategy]['description']}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.mode == "source":
        print(f"\n=== SOURCE QA ({args.strategy}/{args.config}) ===")
        process_source_qa(
            tokenizer, model, device,
            args.baseline_dir, args.output_dir,
            args.config, args.strategy,
            args.max_rows, args.seed
        )

    elif args.mode == "bt":
        print(f"\n=== BT QA ({args.strategy}/{args.lang}/{args.config}) ===")
        process_bt_qa(
            tokenizer, model, device,
            args.baseline_dir, args.contratico_dir, args.output_dir,
            args.config, args.lang, args.strategy,
            args.max_rows, args.seed
        )


if __name__ == "__main__":
    main()
