"""
QA Script for Prompt Ablation Study
Generates answers using different prompt strategies.

Usage:
  python qa_ablation.py --strategy P0-vanilla --mode source --qg_input_path /path/to/qg.jsonl --output_path /path/to/output.jsonl
  python qa_ablation.py --strategy P1-fewshot --mode bt --lang de --qg_input_path /path/to/qg.jsonl --output_path /path/to/output.jsonl
"""

import torch
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import get_prompt, PROMPTS

model_id = "Qwen/Qwen2.5-3B-Instruct"
LANGUAGES = ["de", "es", "fr", "ru", "zh-CN"]


def parse_questions(questions_str):
    """Parse questions from string to list."""
    if isinstance(questions_str, list):
        return questions_str
    if not questions_str or questions_str.strip() == "":
        return []
    try:
        questions = json.loads(questions_str)
        if isinstance(questions, list):
            return questions
        return [str(questions)]
    except json.JSONDecodeError:
        return [questions_str.strip()]


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
            max_new_tokens=50,
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


def generate_answers_for_questions(tokenizer, model, device, sentence, questions_str, strategy):
    """Generate answers for all questions."""
    questions = parse_questions(questions_str)
    
    if not questions:
        return []
    
    answers = []
    for q in questions:
        answer = generate_single_answer(tokenizer, model, device, sentence, q, strategy)
        answers.append(answer)
    
    return answers


def process_source_qa(tokenizer, model, device, qg_file, output_file, strategy):
    """Process source QA with specified strategy."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Collect unique src
    print(f"Collecting unique source sentences for strategy: {strategy}...")
    unique_src = {}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            src = data.get('src', '')
            
            if src not in unique_src:
                unique_src[src] = {
                    'src': src,
                    'lang_tgt': data.get('lang_tgt', ''),
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_src[src]['row_indexes'].append(idx)
    
    print(f"Found {len(unique_src)} unique src values")
    
    # Check for resume
    processed_src = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_src.add(data.get('src', ''))
        print(f"Resuming: {len(processed_src)} already processed")
    
    # Process
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (src, data) in enumerate(unique_src.items()):
            if src in processed_src:
                continue
            
            questions = parse_questions(data['questions'])
            print(f"[{i+1}/{len(unique_src)}] {strategy} - Processing src with {len(questions)} questions...")
            
            answers = generate_answers_for_questions(
                tokenizer, model, device,
                data['src'], data['questions'],
                strategy
            )
            
            if answers:
                print(f"> First answer: {answers[0][:60]}...")
            
            output_row = {
                'src': data['src'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes'],
                'strategy': strategy
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nSource QA ({strategy}) completed. Output: {output_file}")


def process_bt_qa(tokenizer, model, device, qg_file, output_file, lang, strategy):
    """Process BT QA for a specific language with specified strategy."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Collecting unique bt_tgt for language: {lang}, strategy: {strategy}...")
    unique_bt = {}
    
    with open(qg_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            if data.get('lang_tgt', '') != lang:
                continue
            
            src = data.get('src', '')
            bt_tgt = data.get('bt_tgt', '')
            key = (src, bt_tgt)
            
            if key not in unique_bt:
                unique_bt[key] = {
                    'src': src,
                    'bt_tgt': bt_tgt,
                    'lang_tgt': lang,
                    'questions': data.get('questions', ''),
                    'row_indexes': [idx]
                }
            else:
                unique_bt[key]['row_indexes'].append(idx)
    
    total_rows = sum(len(d['row_indexes']) for d in unique_bt.values())
    print(f"Found {len(unique_bt)} unique (src, bt_tgt) pairs from {total_rows} rows")
    
    # Check for resume
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = (data.get('src', ''), data.get('bt_tgt', ''))
                processed_keys.add(key)
        print(f"Resuming: {len(processed_keys)} already processed")
    
    # Process
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, (key, data) in enumerate(unique_bt.items()):
            if key in processed_keys:
                continue
            
            questions = parse_questions(data['questions'])
            print(f"[{i+1}/{len(unique_bt)}] {strategy}/{lang} - {len(questions)} questions...")
            
            answers = generate_answers_for_questions(
                tokenizer, model, device,
                data['bt_tgt'], data['questions'],
                strategy
            )
            
            if answers:
                print(f"> First answer: {answers[0][:60]}...")
            
            output_row = {
                'src': data['src'],
                'bt_tgt': data['bt_tgt'],
                'lang_tgt': data['lang_tgt'],
                'questions': data['questions'],
                'answers': answers,
                'row_indexes': data['row_indexes'],
                'strategy': strategy
            }
            f_out.write(json.dumps(output_row, ensure_ascii=False) + '\n')
    
    print(f"\nBT QA ({strategy}/{lang}) completed. Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="QA Script for Prompt Ablation Study")
    parser.add_argument("--strategy", type=str, required=True, choices=list(PROMPTS.keys()),
                        help=f"Prompt strategy: {list(PROMPTS.keys())}")
    parser.add_argument("--mode", type=str, required=True, choices=["source", "bt"],
                        help="Mode: 'source' or 'bt'")
    parser.add_argument("--lang", type=str, choices=LANGUAGES,
                        help="Language for bt mode")
    parser.add_argument("--qg_input_path", type=str, required=True,
                        help="Path to QG input file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for results")
    args = parser.parse_args()
    
    if args.mode == "bt" and not args.lang:
        parser.error("--lang is required when --mode is 'bt'")
    
    if not os.path.exists(args.qg_input_path):
        print(f"QG file not found: {args.qg_input_path}")
        return
    
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
        print(f"\n=== SOURCE QA ({args.strategy}) ===")
        process_source_qa(tokenizer, model, device, args.qg_input_path, args.output_path, args.strategy)
    
    elif args.mode == "bt":
        print(f"\n=== BT QA ({args.strategy}/{args.lang}) ===")
        process_bt_qa(tokenizer, model, device, args.qg_input_path, args.output_path, args.lang, args.strategy)


if __name__ == "__main__":
    main()
