"""Tests for SentenceBuilder — uses temp dirs, no input/ reads."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.sentence_builder import SentenceBuilder


def _make_cfg(words_path: str, project_root: str, num_sentences: int = 500) -> dict:
    return {
        "words_path": os.path.relpath(words_path, project_root),
        "num_sentences": num_sentences,
        "sentence_min_len": 5,
        "sentence_max_len": 6,
        "dedup_threshold": 0.05,
        "max_dedup_retries": 3,
        "data_seed": 42,
    }


def _write_vocab(path: str, n: int = 200) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"word{i:04d}\n")


class TestSentenceBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name
        self.words_path = os.path.join(self.root, "words.txt")
        _write_vocab(self.words_path, 200)
        os.makedirs(os.path.join(self.root, "output", "logs"), exist_ok=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _builder(self, num_sentences: int = 500) -> SentenceBuilder:
        cfg = _make_cfg(self.words_path, self.root, num_sentences)
        return SentenceBuilder(cfg, self.root)

    def test_build_returns_correct_count(self):
        builder = self._builder(500)
        sentences, _ = builder.build()
        # After dedup, count must be <= num_sentences and > 0
        self.assertGreater(len(sentences), 0)
        self.assertLessEqual(len(sentences), 500)

    def test_sentences_have_valid_length(self):
        builder = self._builder(500)
        sentences, _ = builder.build()
        for s in sentences:
            word_count = len(s.split())
            self.assertIn(
                word_count,
                {5, 6},
                msg=f"Sentence has unexpected length {word_count}: {s!r}",
            )

    def test_no_repeated_words_within_sentence(self):
        builder = self._builder(500)
        sentences, _ = builder.build()
        sample = sentences[:100]
        for s in sample:
            words = s.split()
            self.assertEqual(
                len(words),
                len(set(words)),
                msg=f"Repeated word in sentence: {s!r}",
            )

    def test_dedup_pass_logged(self):
        builder = self._builder(500)
        builder.build()
        log_path = os.path.join(self.root, "output", "logs", "dataset_stats.txt")
        self.assertTrue(os.path.exists(log_path), "Log file not created")
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("PASS", content, "Log does not contain 'PASS'")

    def test_returns_final_seed(self):
        cfg = _make_cfg(self.words_path, self.root, 500)
        data_seed = cfg["data_seed"]
        builder = SentenceBuilder(cfg, self.root)
        _, returned_seed = builder.build()
        self.assertIsInstance(returned_seed, int)
        self.assertGreaterEqual(returned_seed, data_seed)


if __name__ == "__main__":
    unittest.main()
