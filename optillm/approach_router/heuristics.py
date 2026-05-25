"""Post-ML heuristics to correct systematic mis-routing."""

import re
from typing import List, Optional, Tuple

# Algorithm / data-structure design (not symbolic constraint solving)
_ALGORITHM_DESIGN_PATTERNS = [
    re.compile(r"\bO\s*\(\s*n", re.I),
    re.compile(r"\b(time|space)\s+complexit", re.I),
    re.compile(r"\bdesign\s+an?\s+(algorithm|solution|approach)", re.I),
    re.compile(r"\b(algorithm|approach)\s+to\s+(find|solve|compute|implement)", re.I),
    re.compile(r"\bsecond[- ]?(largest|smallest|max|min)", re.I),
    re.compile(
        r"\b(data\s+structure|linked\s+list|binary\s+tree|hash\s+table|unsorted\s+array)\b",
        re.I,
    ),
    re.compile(r"\b(traverse|traversal)\s+(the\s+)?(array|tree|graph|list)\b", re.I),
    re.compile(r"\b(two\s+pointers?|sliding\s+window|dynamic\s+programming)\b", re.I),
]

# Constraint / proof tasks where z3 is appropriate
_Z3_FAVORING_PATTERNS = [
    re.compile(r"\b(integer|natural)\s+solutions?\b", re.I),
    re.compile(r"\b(satisfy|constraints?)\b", re.I),
    re.compile(r"\b(for\s+all|there\s+exists)\b", re.I),
    re.compile(r"\b\d+\s*[xyz]\s*\+\s*\d+\s*[xyz]", re.I),
]

_NON_Z3_PRIORITY = [
    "plansearch",
    "cot_reflection",
    "none",
    "mcts",
    "re2",
    "bon",
    "moa",
    "leap",
    "rstar",
    "self_consistency",
]


def is_algorithm_design_task(text: str) -> bool:
    """True when the prompt is CS algorithm design, not symbolic solving."""
    if not text or not text.strip():
        return False
    if any(p.search(text) for p in _Z3_FAVORING_PATTERNS):
        if not re.search(r"\bO\s*\(", text, re.I):
            return False
    return any(p.search(text) for p in _ALGORITHM_DESIGN_PATTERNS)


def pick_non_z3_approach(
    top_scores: List[Tuple[str, float]],
) -> Tuple[str, float]:
    """Choose best non-z3 label from ML top-k using task-appropriate priority."""
    by_label = {label: score for label, score in top_scores if label != "z3"}
    for preferred in _NON_Z3_PRIORITY:
        if preferred in by_label:
            return preferred, by_label[preferred]
    if by_label:
        label, score = max(by_label.items(), key=lambda item: item[1])
        return label, score
    return "none", 0.0


def apply_z3_algorithm_override(
    approach: str,
    confidence: float,
    top_scores: List[Tuple[str, float]],
    system_prompt: str,
    initial_query: str,
) -> Tuple[str, float, str, Optional[str]]:
    """
    If ML chose z3 for an algorithm-design prompt, reroute.

    Returns (approach, confidence, source, override_reason).
    """
    if approach != "z3":
        return approach, confidence, "ml", None

    text = f"{system_prompt or ''}\n{initial_query or ''}"
    if not is_algorithm_design_task(text):
        return approach, confidence, "ml", None

    new_approach, new_confidence = pick_non_z3_approach(top_scores)
    return (
        new_approach,
        new_confidence,
        "heuristic",
        "algorithm_design_not_z3",
    )
