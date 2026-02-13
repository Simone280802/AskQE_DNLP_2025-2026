"""
Average number of questions per entry in the baseline QG file (biomqm).
Only unique src entries are considered (duplicates are skipped).
Adapted from ner-extension/evaluation/desiderata/i_avg_questions.py for the baseline QG format.
"""
import json
import os

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(os.path.dirname(script_dir))  # baseline/
qg_file = os.path.join(baseline_dir, "QG", "qwen-3b.jsonl")


def extract_question_strings(questions):
    """Extract plain question strings from the QG format.
    Baseline format: '["q1", "q2"]' (JSON string) or ["q1", "q2"] (list of strings)
    NER format: [{"question": "...", "entity_type": "...", "entity_text": "..."}, ...]
    """
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

    seen_srcs = set()
    total_entries = 0
    total_questions = 0
    duplicate_questions_count = 0
    skipped_duplicates = 0

    with open(qg_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            src = data.get("src", "")

            if src in seen_srcs:
                skipped_duplicates += 1
                continue
            seen_srcs.add(src)

            total_entries += 1
            questions = extract_question_strings(data.get("questions", []))

            unique_questions = set(questions)
            if len(unique_questions) < len(questions):
                duplicate_questions_count += 1
            total_questions += len(questions)

    avg_questions = total_questions / total_entries if total_entries > 0 else 0
    print(f"Total unique src entries: {total_entries}")
    print(f"Skipped duplicate src entries: {skipped_duplicates}")
    print(f"Total questions: {total_questions}")
    print(f"Average Number of Questions: {avg_questions:.2f}")


if __name__ == "__main__":
    main()
