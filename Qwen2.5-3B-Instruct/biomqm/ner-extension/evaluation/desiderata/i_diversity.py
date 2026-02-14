"""
Diversity evaluation for NER extension QG file (biomqm).
Computes intra-entry cosine similarity (SBERT) and BERTScore among generated questions.
Adapted from evaluation/desiderata/i_diversity.py for the entity-aware QG format.
"""
import json
import os
import itertools
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import bert_score

# ── Paths ──
script_dir = os.path.dirname(os.path.abspath(__file__))
ner_ext_dir = os.path.dirname(os.path.dirname(script_dir))  # ner-extension/
qg_file = os.path.join(ner_ext_dir, "QG", "qg_entity_aware.jsonl")
output_dir = os.path.join(script_dir, "diversity")
os.makedirs(output_dir, exist_ok=True)

# ── Model ──
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("all-mpnet-base-v2")
sbert_model.to(device)


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
    print(f"\nProcessing File: {qg_file}")

    total_entries = 0
    diversity_scores = []
    processed_data = []

    with open(qg_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                total_entries += 1
                questions = extract_question_strings(data.get("questions", []))

                if len(questions) < 2:
                    continue

                question_pairs = list(itertools.combinations(questions, 2))
                embeddings = sbert_model.encode(questions, convert_to_tensor=True)
                cosine_similarities = [
                    util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                    for i, j in itertools.combinations(range(len(questions)), 2)
                ]

                P, R, F1 = bert_score.score(questions, questions, lang="en", rescale_with_baseline=True)
                bert_similarities = [
                    F1[i].item() for i, j in itertools.combinations(range(len(questions)), 2)
                ]

                avg_cosine_sim = np.mean(cosine_similarities) if cosine_similarities else 0
                avg_bert_sim = np.mean(bert_similarities) if bert_similarities else 0
                diversity_scores.append((avg_cosine_sim, avg_bert_sim))

                data["cosine_similarity"] = avg_cosine_sim
                data["bert_similarity"] = avg_bert_sim
                processed_data.append(data)

            except json.JSONDecodeError as e:
                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                continue

    if diversity_scores:
        avg_sbert_diversity = np.mean([s[0] for s in diversity_scores])
        avg_bert_diversity = np.mean([s[1] for s in diversity_scores])
    else:
        avg_sbert_diversity = 0
        avg_bert_diversity = 0

    print(f"Total entries: {total_entries}")
    print(f"Overall Average Cosine Similarity (SBERT): {avg_sbert_diversity:.4f}")
    print(f"Overall Average BERTScore Similarity: {avg_bert_diversity:.4f}")

    avg_score_entry = {
        "overall_avg_cosine_similarity": avg_sbert_diversity,
        "overall_avg_bert_similarity": avg_bert_diversity
    }
    processed_data.append(avg_score_entry)

    output_file = os.path.join(output_dir, "qg_entity_aware.jsonl")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry in processed_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
