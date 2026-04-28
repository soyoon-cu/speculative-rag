"""Answer normalization and prompt formatting utilities."""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace, remove articles.

    Follows the exact normalization used by Self-RAG and
    the original TriviaQA evaluation script.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def answer_in_response(gold_answers: list[str], model_response: str) -> bool:
    """Return True if any gold answer appears in the normalized model response.

    This is the containment-based metric used by Self-RAG and adopted by
    Speculative RAG for TriviaQA evaluation.
    """
    norm_response = normalize_answer(model_response)
    return any(normalize_answer(ans) in norm_response for ans in gold_answers)
