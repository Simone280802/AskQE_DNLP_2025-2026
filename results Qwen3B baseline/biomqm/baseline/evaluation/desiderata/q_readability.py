"""
Readability evaluation for baseline QG file (biomqm).
Uses Flesch Reading Ease Score to measure question readability.
Adapted from ner-extension/evaluation/desiderata/q_readability.py for the baseline QG format.
"""
import json
import os
import numpy as np
import textstat

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(os.path.dirname(script_dir))  # baseline/
qg_file = os.path.join(baseline_dir, "QG", "qwen-3b.jsonl")


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


def classify_readability(score):
    if score >= 90:
        return "Very Easy (5th grade)"
    elif score >= 80:
        return "Easy (6th grade)"
    elif score >= 70:
        return "Fairly Easy (7th grade)"
    elif score >= 60:
        return "Standard (8th-9th grade)"
    elif score >= 50:
        return "Fairly Difficult (10th-12th grade)"
    elif score >= 30:
        return "Difficult (College)"
    else:
        return "Very Difficult (Graduate level)"


def main():
    print(f"File: {qg_file}")

    total_entries = 0
    readability_scores = []

    with open(qg_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            questions = extract_question_strings(data.get("questions", []))

            if len(questions) == 0:
                continue

            total_entries += 1
            instance_scores = []

            for question in questions:
                score = textstat.flesch_reading_ease(question)
                instance_scores.append(score)

            avg_instance_score = np.mean(instance_scores)
            readability_scores.append(avg_instance_score)

    if readability_scores:
        avg_readability = np.mean(readability_scores)
        print(f"Total entries: {total_entries}")
        print(f"Average Readability Score (Flesch-Kincaid): {avg_readability:.2f}")
        print(f"Division: {classify_readability(avg_readability)}")
    else:
        print("No valid questions found in dataset.")


if __name__ == "__main__":
    main()
