"""Prompt construction for Mistral/Mixtral instruction-tuned models.

Implements the template shown in Appendix I, Figure 8 of:
  Wang et al., "Speculative RAG: Enhancing Retrieval Augmented Generation
  Through Drafting", ICLR 2025.
"""

from __future__ import annotations

_MAX_DOC_CHARS = 1_000


def _truncate(text: str, max_chars: int = _MAX_DOC_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def format_documents(passages: list[str]) -> str:
    """Concatenate retrieved passages into a single numbered block."""
    parts: list[str] = []
    for i, p in enumerate(passages, start=1):
        parts.append(f"[{i}] {_truncate(p)}")
    return "\n\n".join(parts)


def build_prompt(question: str, passages: list[str]) -> str:
    """Return the full [INST]…[/INST] prompt for Mistral/Mixtral.

    Template (Appendix I, Figure 8):

        [INST] Given the following documents, answer the question.
        Documents: {retrieved_docs_concatenated}
        Question: {question} [/INST]
    """
    docs_block = format_documents(passages)
    return (
        "[INST] Given the following documents, answer the question.\n"
        f"Documents:\n{docs_block}\n\n"
        f"Question: {question} [/INST]"
    )
