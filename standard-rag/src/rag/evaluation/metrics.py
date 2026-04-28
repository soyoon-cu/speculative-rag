from __future__ import annotations

from dataclasses import dataclass, field

from rag.data.preprocess import answer_in_response


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def update(
        self,
        *,
        question_id: str,
        question: str,
        gold_answers: list[str],
        model_response: str,
    ) -> bool:
        """Record one prediction. Returns True if the answer is correct."""
        correct = answer_in_response(gold_answers, model_response)
        self.total += 1
        if correct:
            self.correct += 1
        self.details.append(
            {
                "question_id": question_id,
                "question": question,
                "gold_answers": gold_answers,
                "model_response": model_response,
                "correct": correct,
            }
        )
        return correct

    def summary(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy * 100, 2),
        }
