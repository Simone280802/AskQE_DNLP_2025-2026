"""
Entity-Aware QA for contraTICO BT sentences

Answers NER entity-aware questions on contraTICO perturbed text.
Processes one perturbation file at a time, matching by ID with QG output.
Outputs combined source + BT answers for direct evaluation.

Usage:
    python qa_entity_contratico.py \
        --qg_path /path/to/qg_entity_aware.jsonl \
        --qa_source_path /path/to/source.jsonl \
        --contratico_path /path/to/contratico/en-es/alteration.jsonl \
        --lang es \
        --output_path /path/to/output.jsonl \
        [--sample_size 84] [--seed 42] [--max_samples N]
"""

import json
import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

QA_PROMPT = """Task: Answer the question based on the given sentence.

Sentence: {sentence}
Question: {question}

Instructions:
- Answer using ONLY information from the sentence
- If the answer is not in the sentence, respond with "[NOT FOUND]"
- Be concise and direct

Answer:"""


def generate_answer(tokenizer, model, device, sentence, question):
    """Generate an answer for a question about the sentence."""
    prompt = QA_PROMPT.format(sentence=sentence, question=question)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on given text."},
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
            max_new_tokens=64,
            temperature=0.3,
            do_sample=True,
        )

    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="Entity-Aware QA for contraTICO BT")
    parser.add_argument("--qg_path", type=str, required=True,
                        help="Path to entity-aware QG output JSONL")
    parser.add_argument("--qa_source_path", type=str, required=True,
                        help="Path to QA source output JSONL")
    parser.add_argument("--contratico_path", type=str, required=True,
                        help="Path to original contraTICO perturbation JSONL file")
    parser.add_argument("--lang", type=str, required=True,
                        choices=["es", "fr", "hi", "tl", "zh"],
                        help="Target language")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--sample_size", type=int, default=125,
                        help="Number of rows to sample (default: 84)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for testing")
    args = parser.parse_args()

    # ── Load QG data (indexed by id) ──
    qg_by_id = {}
    with open(args.qg_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            qg_by_id[data['id']] = data
    print(f"Loaded {len(qg_by_id)} QG entries")

    # ── Load QA source answers (indexed by id) ──
    src_by_id = {}
    with open(args.qa_source_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            src_by_id[data.get('id', data.get('src', ''))] = data
    print(f"Loaded {len(src_by_id)} source QA entries")

    # ── Load and sample contraTICO data ──
    all_rows = []
    with open(args.contratico_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    n_sample = min(args.sample_size, len(all_rows))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(all_rows)), n_sample))
    sampled = [all_rows[i] for i in indices]

    print(f"Sampled {len(sampled)} rows from {os.path.basename(args.contratico_path)}")

    if args.max_samples:
        sampled = sampled[:args.max_samples]

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)

    # ── Process ──
    pert_field = f"pert_{args.lang}"
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    processed = 0
    skipped = 0

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for row in sampled:
            row_id = row.get('id', '')
            qg_data = qg_by_id.get(row_id)
            src_data = src_by_id.get(row_id)

            if not qg_data:
                skipped += 1
                continue

            processed += 1
            if processed % 20 == 0:
                print(f"[{processed}/{len(sampled)}] Processing BT {args.lang}...")

            pert_text = row.get(pert_field, '')
            questions = qg_data.get('questions', [])

            # Answer questions on perturbed text
            answers_bt = []
            for q_info in questions:
                question = q_info.get('question', '')
                answer = generate_answer(tokenizer, model, device, pert_text, question)
                answers_bt.append({
                    'question': question,
                    'entity_type': q_info.get('entity_type', 'UNKNOWN'),
                    'entity_text': q_info.get('entity_text', ''),
                    'answer': answer,
                })

            # Get source answers
            answers_src = src_data.get('answers', []) if src_data else []

            output_row = {
                'id': row_id,
                'src': row.get('en', ''),
                'bt_tgt': pert_text,
                'lang_tgt': args.lang,
                'perturbation': row.get('perturbation', ''),
                'questions': questions,
                'answers_src': answers_src,
                'answers_bt': answers_bt,
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 50}")
    print(f"QA BT Complete ({args.lang})")
    print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {args.output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
