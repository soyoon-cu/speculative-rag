"""Unit tests for prompt construction and evaluation metrics."""

from __future__ import annotations

import pytest

from rag.evaluation.metrics import EvalResult
from rag.generation.prompts import _MAX_DOC_CHARS, _truncate, build_prompt, format_documents


class TestTruncate:
    def test_short_text_unchanged(self):
        text = "short"
        assert _truncate(text) == text

    def test_exact_limit_unchanged(self):
        text = "a" * _MAX_DOC_CHARS
        assert _truncate(text) == text

    def test_long_text_truncated(self):
        text = "word " * 300
        result = _truncate(text)
        assert len(result) <= _MAX_DOC_CHARS + 4

    def test_truncates_at_word_boundary(self):
        text = "hello " * 300
        result = _truncate(text)
        assert not result.rstrip(" .").endswith("hell")

    def test_truncated_ends_with_ellipsis(self):
        text = "word " * 300
        assert _truncate(text).endswith(" ...")


class TestFormatDocuments:
    def test_numbered_passages(self):
        result = format_documents(["first", "second", "third"])
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_passage_content_included(self):
        result = format_documents(["alpha passage", "beta passage"])
        assert "alpha passage" in result
        assert "beta passage" in result

    def test_single_passage(self):
        result = format_documents(["only one"])
        assert "[1]" in result
        assert "[2]" not in result

    def test_empty_passages(self):
        assert format_documents([]) == ""


class TestBuildPrompt:
    def test_contains_inst_tags(self):
        prompt = build_prompt("Who wrote Hamlet?", ["Shakespeare was a playwright."])
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_contains_question(self):
        question = "What is the capital of France?"
        prompt = build_prompt(question, ["Paris is in France."])
        assert question in prompt

    def test_contains_documents(self):
        passages = ["Doc A content.", "Doc B content."]
        prompt = build_prompt("Q?", passages)
        assert "Doc A content." in prompt
        assert "Doc B content." in prompt

    def test_document_numbering(self):
        passages = ["first", "second"]
        prompt = build_prompt("Q?", passages)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_question_label_present(self):
        prompt = build_prompt("What year?", ["Some context."])
        assert "Question:" in prompt

    def test_documents_label_present(self):
        prompt = build_prompt("Q?", ["ctx"])
        assert "Documents:" in prompt

    def test_truncates_long_documents(self):
        long_doc = "word " * 500
        prompt = build_prompt("Q?", [long_doc])
        assert "[/INST]" in prompt

    def test_question_after_documents(self):
        prompt = build_prompt("My question?", ["passage"])
        assert prompt.index("Documents:") < prompt.index("Question:")


class TestEvalResult:
    def test_accuracy_zero_when_empty(self):
        result = EvalResult()
        assert result.accuracy == 0.0

    def test_correct_counting(self):
        result = EvalResult()
        result.update(
            question_id="q1",
            question="Who wrote Hamlet?",
            gold_answers=["Shakespeare"],
            model_response="Shakespeare wrote Hamlet.",
        )
        result.update(
            question_id="q2",
            question="What is 2+2?",
            gold_answers=["4"],
            model_response="The answer is 5.",
        )
        assert result.total == 2
        assert result.correct == 1
        assert result.accuracy == pytest.approx(0.5)

    def test_update_returns_bool(self):
        result = EvalResult()
        assert (
            result.update(
                question_id="q1", question="Q", gold_answers=["yes"], model_response="yes"
            )
            is True
        )
        assert (
            result.update(question_id="q2", question="Q", gold_answers=["yes"], model_response="no")
            is False
        )

    def test_details_populated(self):
        result = EvalResult()
        result.update(
            question_id="q1",
            question="Who?",
            gold_answers=["Alice"],
            model_response="Alice did it.",
        )
        assert len(result.details) == 1
        entry = result.details[0]
        assert entry["question_id"] == "q1"
        assert entry["correct"] is True
        assert entry["model_response"] == "Alice did it."

    def test_summary_keys(self):
        result = EvalResult()
        assert {"total", "correct", "accuracy"} <= result.summary().keys()

    def test_summary_accuracy_value(self):
        result = EvalResult()
        result.update(
            question_id="q1", question="Q", gold_answers=["Paris"], model_response="Paris"
        )
        result.update(
            question_id="q2", question="Q", gold_answers=["Paris"], model_response="Berlin"
        )
        assert result.summary()["accuracy"] == pytest.approx(50.0)

    def test_all_correct(self):
        result = EvalResult()
        for i in range(5):
            result.update(
                question_id=str(i), question="Q", gold_answers=["ans"], model_response="ans"
            )
        assert result.accuracy == pytest.approx(1.0)

    def test_none_correct(self):
        result = EvalResult()
        for i in range(3):
            result.update(
                question_id=str(i), question="Q", gold_answers=["ans"], model_response="wrong"
            )
        assert result.correct == 0
        assert result.accuracy == pytest.approx(0.0)
