"""
Duplicate question detection for NER extension QG file (biomqm).
Adapted from evaluation/desiderata/i_duplicate.py for the entity-aware QG format.
"""
import json
import os

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
ner_ext_dir = os.path.dirname(os.path.dirname(script_dir))  # ner-extension/
qg_file = os.path.join(ner_ext_dir, "QG", "qg_entity_aware.jsonl")


def extract_question_strings(questions):
    """Extract plain question strings from the NER QG format."""
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
    print(f"File: {qg_file}")

    total_entries = 0
    duplicate_questions_count = 0

    with open(qg_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            total_entries += 1
            questions = extract_question_strings(data.get("questions", []))

            unique_questions = set(questions)
            if len(unique_questions) < len(questions):
                duplicate_questions_count += 1

    print(f"Duplicate Questions: {duplicate_questions_count} / {total_entries}")


if __name__ == "__main__":
    main()
