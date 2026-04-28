"""Load and iterate over TriviaQA (rc.wikipedia split) from HuggingFace."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from datasets import Dataset, load_dataset


@dataclass
class TriviaQASample:
    question_id: str
    question: str
    answers: list[str]


def load_triviaqa(split: str = "validation") -> Dataset:
    """Return the raw HuggingFace Dataset for TriviaQA rc.wikipedia.

    Args:
        split: One of "train", "validation", or "test".
               Use "validation" for eval (test labels are withheld on the leaderboard).
    """
    return load_dataset("trivia_qa", "rc.wikipedia", split=split, trust_remote_code=True)


def iter_samples(split: str = "validation") -> Iterator[TriviaQASample]:
    """Yield TriviaQASample objects for every example in the split."""
    ds = load_triviaqa(split)
    for row in ds:
        answer_field = row["answer"]
        gold_answers: list[str] = [answer_field["value"]] + list(answer_field.get("aliases", []))
        seen: set[str] = set()
        unique_answers: list[str] = []
        for a in gold_answers:
            if a not in seen:
                seen.add(a)
                unique_answers.append(a)
        yield TriviaQASample(
            question_id=row["question_id"],
            question=row["question"],
            answers=unique_answers,
        )
