"""Verify consistent lexical normalization for natural-language search."""

import unittest

from ir.preprocessing import tokenize


class PreprocessingTests(unittest.TestCase):
    def test_removes_stopwords_and_lemmatizes_word_forms(self) -> None:
        self.assertEqual(
            tokenize("The bosses are running through fantasy worlds"),
            ["boss", "run", "fantasy", "world"],
        )


if __name__ == "__main__":
    unittest.main()
