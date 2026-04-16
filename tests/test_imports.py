"""Smoke-test: verify all project modules import cleanly."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImports(unittest.TestCase):

    def test_import_sentence_builder(self):
        from src.data.sentence_builder import SentenceBuilder
        self.assertTrue(callable(SentenceBuilder))

    def test_import_dataset_builder(self):
        from src.data.dataset import DatasetBuilder
        self.assertTrue(callable(DatasetBuilder))

    def test_import_sequence_model(self):
        from src.models.rnn_model import SequenceModel
        self.assertTrue(callable(SequenceModel))

    def test_import_trainer(self):
        from src.training.trainer import Trainer
        self.assertTrue(callable(Trainer))

    def test_import_run_evaluation(self):
        from src.evaluation.evaluator import run_evaluation
        self.assertTrue(callable(run_evaluation))

    def test_import_get_device(self):
        from src.utils.device import get_device
        self.assertTrue(callable(get_device))


if __name__ == "__main__":
    unittest.main()
