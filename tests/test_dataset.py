"""Tests for DatasetBuilder — synthetic in-memory sentences, no input/ reads."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import DatasetBuilder


_VOCAB_50 = [f"tok{i:03d}" for i in range(50)]


def _make_sentences(n: int = 100, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n):
        length = rng.choice([5, 6])
        words = rng.sample(_VOCAB_50, length)
        sentences.append(" ".join(words))
    return sentences


def _make_cfg() -> dict:
    return {
        "train_ratio": 0.8,
        "test_ratio": 0.1,
        "sentence_min_len": 5,
    }


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.sentences = _make_sentences(100, seed=42)

    def _builder(self, predict_n: int = 2, sentences=None) -> DatasetBuilder:
        sents = sentences if sentences is not None else self.sentences
        return DatasetBuilder(
            sentences=sents,
            final_seed=42,
            predict_n=predict_n,
            data_cfg=_make_cfg(),
            batch_size=8,
        )

    def test_vocab_from_train_only(self):
        builder = self._builder()
        _, _, _, vocab = builder.build()

        # Reconstruct train split with same seed to verify vocab matches train tokens
        shuffled = self.sentences[:]
        random.seed(42)
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * 0.8)
        train_sents = shuffled[:train_end]

        expected_tokens: set[str] = set()
        for s in train_sents:
            expected_tokens.update(s.split())
        expected_vocab_keys = {"<PAD>", "<UNK>"} | expected_tokens
        self.assertEqual(set(vocab.keys()), expected_vocab_keys)

    def test_split_sizes(self):
        builder = self._builder()
        train_loader, eval_loader, test_loader, _ = builder.build()
        n = len(self.sentences)
        train_exp = int(n * 0.8)
        test_exp = int(n * 0.1)
        eval_exp = n - train_exp - test_exp

        self.assertAlmostEqual(len(train_loader.dataset), train_exp, delta=1)
        self.assertAlmostEqual(len(test_loader.dataset), test_exp, delta=1)
        self.assertAlmostEqual(len(eval_loader.dataset), eval_exp, delta=1)

    def test_input_target_shapes(self):
        # predict_n=2: input_len = original_len - 2, target_len = 2
        builder = self._builder(predict_n=2)
        train_loader, _, _, _ = builder.build()
        for input_ids, targets in train_loader:
            self.assertEqual(targets.shape[1], 2)
            # Shortest possible input is sentence_min_len(5) - predict_n(2) = 3
            self.assertGreaterEqual(input_ids.shape[1], 3)
            break

    def test_padding_in_collate(self):
        # Mix 5-word and 6-word sentences to force padding within a batch
        mixed = [
            "tok001 tok002 tok003 tok004 tok005",
            "tok006 tok007 tok008 tok009 tok010 tok011",
            "tok012 tok013 tok014 tok015 tok016",
            "tok017 tok018 tok019 tok020 tok021 tok022",
        ] * 10
        builder = DatasetBuilder(
            sentences=mixed,
            final_seed=0,
            predict_n=1,
            data_cfg=_make_cfg(),
            batch_size=4,
        )
        train_loader, _, _, _ = builder.build()
        for input_ids, _ in train_loader:
            lengths = [input_ids[i].shape[0] for i in range(input_ids.shape[0])]
            self.assertEqual(len(set(lengths)), 1, "Batch inputs must be uniformly padded")
            break

    def test_assert_min_len_raised(self):
        with self.assertRaises(AssertionError):
            DatasetBuilder(
                sentences=self.sentences,
                final_seed=42,
                predict_n=5,        # equals sentence_min_len=5 → must raise
                data_cfg=_make_cfg(),
                batch_size=8,
            )


if __name__ == "__main__":
    unittest.main()
