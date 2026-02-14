"""
Shared utility functions for NER Extension evaluation (ContraTICO).
"""

import re
import string
from collections import Counter
import sacrebleu


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """Get word tokens."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(prediction, ground_truth):
    """Compute exact match (after normalization)."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compare_answers(prediction, ground_truth):
    """
    Compare prediction to ground truth and return metrics.
    
    Returns:
        tuple: (f1, exact_match, bleu, chrf)
    """
    f1 = compute_f1(prediction, ground_truth)
    em = compute_exact_match(prediction, ground_truth)
    
    # Calculate BLEU and chrF using sacrebleu
    # sacrebleu expects list of references for each hypothesis
    bleu_score = sacrebleu.sentence_bleu(prediction, [ground_truth]).score
    chrf_score = sacrebleu.sentence_chrf(prediction, [ground_truth]).score
    
    # Normalize to 0-1 range for consistency with F1/EM if desired, 
    # but standard is 0-100. Let's keep 0-100 or 0-1? 
    # BioMQM usually keeps 0-100 for BLEU/chrF. 
    # However, F1/EM are 0-1. Let's return 0-100 for BLEU/chrF to be standard.
    # actually, seeing broadly used metrics, let's divide by 100 to keep 0-1 scale 
    # consistent with F1/EM in the summary
    
    return f1, em, bleu_score / 100.0, chrf_score / 100.0
