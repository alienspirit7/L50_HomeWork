"""Tests for SequenceModel — in-memory only, no file I/O."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.rnn_model import SequenceModel

# Small dimensions for speed
_VOCAB = 100
_EMBED = 16
_HIDDEN = 32
_LAYERS = 2
_DROPOUT = 0.1
_BATCH = 4
_SEQ = 8


def _model(architecture: str = "rnn", predict_n: int = 1, num_layers: int = _LAYERS) -> SequenceModel:
    return SequenceModel(
        vocab_size=_VOCAB,
        embedding_dim=_EMBED,
        hidden_size=_HIDDEN,
        num_layers=num_layers,
        dropout=_DROPOUT,
        architecture=architecture,
        predict_n=predict_n,
    )


def _inputs(batch: int = _BATCH, seq: int = _SEQ) -> torch.Tensor:
    return torch.randint(1, _VOCAB, (batch, seq))


class TestSequenceModel(unittest.TestCase):

    def test_output_shape_rnn_predict1(self):
        model = _model("rnn", predict_n=1)
        out = model(_inputs())
        self.assertEqual(out.shape, (_BATCH, 1, _VOCAB))

    def test_output_shape_rnn_predict3(self):
        model = _model("rnn", predict_n=3)
        out = model(_inputs())
        self.assertEqual(out.shape, (_BATCH, 3, _VOCAB))

    def test_output_shape_lstm_predict2(self):
        model = _model("lstm", predict_n=2)
        out = model(_inputs())
        self.assertEqual(out.shape, (_BATCH, 2, _VOCAB))

    def test_invalid_architecture_raises(self):
        with self.assertRaises(ValueError):
            _model("transformer", predict_n=1)

    def test_single_layer_no_dropout(self):
        # PyTorch sets dropout=0 for num_layers=1; model must instantiate and run cleanly
        try:
            model = SequenceModel(
                vocab_size=_VOCAB,
                embedding_dim=_EMBED,
                hidden_size=_HIDDEN,
                num_layers=1,
                dropout=0.5,
                architecture="rnn",
                predict_n=1,
            )
            out = model(_inputs())
            self.assertEqual(out.shape, (_BATCH, 1, _VOCAB))
        except Exception as exc:
            self.fail(f"Single-layer model raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
