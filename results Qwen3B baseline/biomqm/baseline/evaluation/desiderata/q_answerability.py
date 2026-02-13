"""
Answerability evaluation for baseline QG file (biomqm).
Uses potsawee/longformer-large-4096-answerable-squad2 to score each question.
Adapted from ner-extension/evaluation/desiderata/q_answerability.py for the baseline QG format.
"""
import json
import os
import numpy as np
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Model ──
model_name = "potsawee/longformer-large-4096-answerable-squad2"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name)
model.to(device)

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(os.path.dirname(script_dir))  # baseline/
qg_file = os.path.join(baseline_dir, "QG", "qwen-3b.jsonl")
output_dir = os.path.join(script_dir, "answerability")
os.makedirs(output_dir, exist_ok=True)


def extract_question_strings(questions):
    """Extract plain question strings from the QG format."""
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except (json.JSONDecodeError, ValueError):
            return []

    if not isinstance(questions, list):
        return []

    result = []
    for q in questions:
        if isinstance(q, dict):
            result.append(q.get("question", ""))
        elif isinstance(q, str):
            result.append(q)
        elif isinstance(q, list):
            result.append(str(q))
    return [r for r in result if r]


def main():
    print(f"\nProcessing File: {qg_file}")

    answerability_scores = []
    total_questions = 0
    processed_data = []

    with open(qg_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                context = data.get("src", "")
                questions = extract_question_strings(data.get("questions", []))

                if not context or not questions:
                    continue

                instance_scores = []
                question_scores = []

                for question in questions:
                    input_text = question + ' ' + tokenizer.sep_token + ' ' + context

                    inputs = tokenizer(input_text, max_length=4096, truncation=True, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    prob = torch.sigmoid(model(**inputs).logits.squeeze(-1))
                    answerability = prob.item() * 100
                    instance_scores.append(answerability)
                    total_questions += 1
                    question_scores.append({"question": question, "answerability_score": answerability})

                if instance_scores:
                    avg_instance_score = np.mean(instance_scores)
                    answerability_scores.append(avg_instance_score)

                data["answerability_scores"] = avg_instance_score
                data["answerability_avg"] = avg_instance_score
                processed_data.append(data)

            except json.JSONDecodeError as e:
                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                continue

    output_file = os.path.join(output_dir, "qwen-3b.jsonl")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry in processed_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if answerability_scores:
        avg_answerability = np.mean(answerability_scores)
        print(f"\nTotal questions scored: {total_questions}")
        print(f"Average Answerability Score: {avg_answerability:.2f}%")
    else:
        print("\nNo valid questions found in dataset.")

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
