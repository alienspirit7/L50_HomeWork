"""Tests for run_evaluation — tiny model + synthetic DataLoader, temp dirs."""
import csv
import os
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.evaluator import run_evaluation
from src.models.rnn_model import SequenceModel

_VOCAB = 20
_EMBED = 8
_HIDDEN = 16
_PREDICT_N = 1
_BATCH = 4
_N_SAMPLES = 8
_VARIANT = "test_variant"


def _tiny_model() -> SequenceModel:
    return SequenceModel(
        vocab_size=_VOCAB,
        embedding_dim=_EMBED,
        hidden_size=_HIDDEN,
        num_layers=1,
        dropout=0.0,
        architecture="rnn",
        predict_n=_PREDICT_N,
    )


def _tiny_loader() -> DataLoader:
    inputs = torch.randint(1, _VOCAB, (_N_SAMPLES, 6))
    targets = torch.randint(1, _VOCAB, (_N_SAMPLES, _PREDICT_N))
    ds = TensorDataset(inputs, targets)
    return DataLoader(ds, batch_size=_BATCH)


def _make_config(output_dir: str) -> dict:
    return {
        "variants": {
            _VARIANT: {
                "architecture": "rnn",
                "predict_n": _PREDICT_N,
            }
        },
        "paths": {
            "output_dir": output_dir,
        },
    }


def _trainer_stats() -> dict:
    return {
        "best_eval_loss": 1.23,
        "best_eval_loss_epoch": 2,
        "total_epochs": 5,
        "data_seed": 42,
        "model_seed": 99,
    }


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmpdir.name
        self.config = _make_config(self.output_dir)
        self.device = torch.device("cpu")
        self.model = _tiny_model()
        self.loader = _tiny_loader()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self):
        return run_evaluation(
            model=self.model,
            test_loader=self.loader,
            device=self.device,
            variant_id=_VARIANT,
            config=self.config,
            trainer_stats=_trainer_stats(),
        )

    def test_metrics_keys(self):
        metrics = self._run()
        required = {
            "variant_id", "architecture", "predict_n", "test_loss",
            "test_seq_acc", "best_eval_loss", "best_eval_loss_epoch",
            "total_epochs", "model_params", "data_seed", "model_seed",
            "run_timestamp",
        }
        self.assertEqual(required, set(metrics.keys()))

    def test_results_csv_created(self):
        self._run()
        csv_path = os.path.join(self.output_dir, "analysis", "results_summary.csv")
        self.assertTrue(os.path.exists(csv_path), "results_summary.csv not created")

    def test_results_csv_appends(self):
        self._run()
        self._run()
        csv_path = os.path.join(self.output_dir, "analysis", "results_summary.csv")
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
        # 1 header row + 2 data rows
        self.assertEqual(len(rows), 3, f"Expected 3 rows (1 header + 2 data), got {len(rows)}")

    def test_seq_acc_range(self):
        metrics = self._run()
        self.assertGreaterEqual(metrics["test_seq_acc"], 0.0)
        self.assertLessEqual(metrics["test_seq_acc"], 1.0)

    def test_model_params_positive(self):
        metrics = self._run()
        self.assertGreater(metrics["model_params"], 0)


if __name__ == "__main__":
    unittest.main()
