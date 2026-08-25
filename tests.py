"""
Tests for the LCEL pipeline.

    python3 tests.py

The parser tests need no model. The optional assignment test — "the decomposer
returns at most 3 sub-questions" — is checked twice over: once against fake model
output for every awkward format, and once against the live model if Ollama is up.
"""

import unittest

from lcel_pipeline import parse_answer_block, parse_numbered_subquestions


class FakeMessage:
    """Stands in for a BaseMessage, which only needs a .content attribute here."""

    def __init__(self, content):
        self.content = content


class TestDecomposerCap(unittest.TestCase):
    """The assignment's requirement: never more than 3 sub-questions."""

    def test_three_items_stay_three(self):
        reply = FakeMessage("1. First?\n2. Second?\n3. Third?")
        self.assertEqual(len(parse_numbered_subquestions(reply)), 3)

    def test_a_model_returning_six_is_capped_at_three(self):
        """The prompt says 'up to 3', but a prompt is a request, not a guarantee."""
        reply = FakeMessage("\n".join(f"{n}. Question {n}?" for n in range(1, 7)))
        result = parse_numbered_subquestions(reply)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Question 1?")
        self.assertEqual(result[-1], "Question 3?")

    def test_never_exceeds_three_on_any_shape(self):
        shapes = [
            "1. a\n2. b\n3. c\n4. d",
            "1) a\n2) b\n3) c\n4) d",
            "  1.  a\n  2.  b\n  3.  c\n  4.  d",
            "1. a\n\n2. b\n\n3. c\n\n4. d\n\n5. e",
        ]
        for text in shapes:
            self.assertLessEqual(len(parse_numbered_subquestions(FakeMessage(text))), 3, text)


class TestSubquestionParsing(unittest.TestCase):
    def test_accepts_both_dot_and_paren(self):
        self.assertEqual(
            parse_numbered_subquestions(FakeMessage("1) First?\n2) Second?")),
            ["First?", "Second?"],
        )

    def test_ignores_preamble_lines(self):
        reply = FakeMessage("Here are the sub-questions:\n1. Real one?\n2. Another?")
        self.assertEqual(parse_numbered_subquestions(reply), ["Real one?", "Another?"])

    def test_unnumbered_text_becomes_one_subquestion(self):
        """The fallback the brief asks for: never return an empty list."""
        result = parse_numbered_subquestions(FakeMessage("Just one sentence, no numbering."))
        self.assertEqual(result, ["Just one sentence, no numbering."])

    def test_empty_reply_gives_an_empty_list(self):
        self.assertEqual(parse_numbered_subquestions(FakeMessage("   ")), [])

    def test_works_on_a_plain_string_too(self):
        """getattr(msg, 'content', str(msg)) means a bare string still parses."""
        self.assertEqual(parse_numbered_subquestions("1. From a string?"), ["From a string?"])


class TestAnswerParsing(unittest.TestCase):
    def test_extracts_answer_and_steps(self):
        parsed = parse_answer_block("Answer: Use a cache.\nSteps:\n- Measure first\n- Add Redis")

        self.assertEqual(parsed["answer"], "Use a cache.")
        self.assertEqual(parsed["steps"], ["Measure first", "Add Redis"])

    def test_accepts_bullet_variants(self):
        parsed = parse_answer_block("Answer: X\nSteps:\n• one\n* two\n- three")
        self.assertEqual(parsed["steps"], ["one", "two", "three"])

    def test_case_insensitive_answer_label(self):
        self.assertEqual(parse_answer_block("ANSWER: Yes\n- step").get("answer"), "Yes")

    def test_missing_label_falls_back_to_the_whole_text(self):
        parsed = parse_answer_block("The model ignored the format entirely.")
        self.assertEqual(parsed["answer"], "The model ignored the format entirely.")
        self.assertEqual(parsed["steps"], ["(no steps parsed)"])

    def test_raw_text_is_always_kept(self):
        raw = "Answer: X\nSteps:\n- y"
        self.assertEqual(parse_answer_block(raw)["raw"], raw)


class TestAgainstTheLiveModel(unittest.TestCase):
    """Same cap, checked end to end. Skipped when Ollama is not running."""

    @classmethod
    def setUpClass(cls):
        from lcel_pipeline import decomposer, parse_subq_runnable

        cls.chain = decomposer | parse_subq_runnable

        try:
            cls.result = cls.chain.invoke(
                {"question": "How do I make a slow Django app faster?"}
            )
        except Exception as error:  # noqa: BLE001 - any connection problem skips
            raise unittest.SkipTest(f"Ollama unavailable: {type(error).__name__}")

    def test_returns_at_most_three(self):
        self.assertLessEqual(len(self.result), 3)

    def test_returns_at_least_one(self):
        self.assertGreaterEqual(len(self.result), 1)

    def test_every_subquestion_is_a_non_empty_string(self):
        for item in self.result:
            self.assertIsInstance(item, str)
            self.assertTrue(item.strip())

    def test_no_numbering_survives_the_parse(self):
        """'1. ' must be stripped, or it ends up inside the next prompt."""
        import re

        for item in self.result:
            self.assertIsNone(re.match(r"^\s*\d+\s*[.)]", item), item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
