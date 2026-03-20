"""Unit tests for answer normalization and containment metric."""

from rag.data.preprocess import answer_in_response, normalize_answer


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Albert Einstein") == "albert einstein"

    def test_strips_punctuation(self):
        assert normalize_answer("World War II.") == "world war ii"

    def test_removes_articles(self):
        assert normalize_answer("The United States") == "united states"
        assert normalize_answer("A quick fox") == "quick fox"
        assert normalize_answer("An apple") == "apple"

    def test_collapses_whitespace(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_answer("The   U.S.A.") == "usa"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_numbers_preserved(self):
        assert normalize_answer("1942") == "1942"

    def test_only_punctuation(self):
        assert normalize_answer("...") == ""

    def test_article_not_stripped_mid_word(self):
        assert "ather" in normalize_answer("ather")


class TestAnswerInResponse:
    def test_exact_match(self):
        assert answer_in_response(["Paris"], "The answer is Paris.")

    def test_case_insensitive(self):
        assert answer_in_response(["paris"], "The capital is Paris")

    def test_any_alias_matches(self):
        assert answer_in_response(["NYC", "New York City"], "I love New York City.")

    def test_no_match(self):
        assert not answer_in_response(["Berlin"], "The answer is Paris.")

    def test_punctuation_ignored(self):
        assert answer_in_response(["U.S.A."], "It is the USA.")

    def test_article_stripped(self):
        assert answer_in_response(["The Beatles"], "They were the beatles.")

    def test_empty_response(self):
        assert not answer_in_response(["Paris"], "")

    def test_empty_gold(self):
        assert not answer_in_response([], "Paris is the answer")

    def test_multi_word_answer_contained(self):
        assert answer_in_response(["New York"], "It happened in New York City.")

    def test_first_alias_wrong_second_correct(self):
        assert answer_in_response(["London", "Paris"], "The answer is Paris.")

    def test_all_aliases_wrong(self):
        assert not answer_in_response(["London", "Berlin"], "The answer is Paris.")
