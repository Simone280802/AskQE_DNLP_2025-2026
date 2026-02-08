"""
Prompt Templates for Ablation Study
3 strategies: Few-Shot, Chain-of-Thought, Concise
"""

# =============================================
# P1: FEW-SHOT MEDICAL
# Target: Better domain understanding
# =============================================
P1_FEWSHOT = """Answer medical questions based on the context.

Example 1:
Context: "Patient diagnosed with diabetes mellitus type 2, prescribed metformin 500mg twice daily."
Question: "What medication was prescribed?"
Answer: "metformin 500mg"

Example 2:
Context: "MRI revealed a 2cm lesion in the left frontal lobe."
Question: "What was the size of the lesion?"
Answer: "2cm"

Now answer:
Context: {sentence}
Question: {question}"""


# =============================================
# P2: CHAIN-OF-THOUGHT
# Target: Better reasoning, severity discrimination
# =============================================
P2_COT = """Answer the question about the medical text.

Context: {sentence}
Question: {question}

Think step by step:
1. Identify the key medical information in the context
2. Locate the specific answer to the question
3. Provide the exact answer from the text

Format your response as:
Reasoning: [your step-by-step reasoning]
FINAL ANSWER: [concise answer, max 10 words, using exact terms from context]

Answer:"""


# =============================================
# P3: CONCISE CONSTRAINT
# Target: Reduce verbosity, improve exact match
# =============================================
P3_CONCISE = """Context: {sentence}
Question: {question}

Answer in maximum 5 words using exact terms from the context:"""


# =============================================
# PROMPT REGISTRY
# =============================================
PROMPTS = {
    "P1-fewshot": {
        "template": P1_FEWSHOT,
        "description": "Few-shot with medical examples"
    },
    "P2-cot": {
        "template": P2_COT,
        "description": "Chain-of-thought reasoning"
    },
    "P3-concise": {
        "template": P3_CONCISE,
        "description": "Concise answer constraint"
    }
}


def get_prompt(strategy: str) -> str:
    """Get prompt template for a strategy."""
    if strategy not in PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[strategy]["template"]


def list_strategies():
    """List all available strategies."""
    for name, info in PROMPTS.items():
        print(f"{name}: {info['description']}")
