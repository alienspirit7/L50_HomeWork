import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """RNN or LSTM sequence model for next-token(s) prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        architecture: str,
        predict_n: int,
    ) -> None:
        super().__init__()

        if architecture not in ("rnn", "lstm"):
            raise ValueError(
                f"architecture must be 'rnn' or 'lstm', got '{architecture}'"
            )

        self.predict_n = predict_n
        self.architecture = architecture

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if architecture == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )

        self.output_head = nn.Linear(hidden_size, vocab_size * predict_n)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, predict_n, vocab_size)
        """
        vocab_size = self.output_head.out_features // self.predict_n

        x = self.embedding(input_ids)          # (batch, seq_len, embedding_dim)
        out, _ = self.recurrent(x)             # (batch, seq_len, hidden_size)
        last = out[:, -1, :]                   # (batch, hidden_size)
        logits = self.output_head(last)        # (batch, vocab_size * predict_n)
        return logits.view(-1, self.predict_n, vocab_size)  # (batch, predict_n, vocab_size)
