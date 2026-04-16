import random
import torch
from torch.utils.data import Dataset, DataLoader


class _SentenceDataset(Dataset):
    def __init__(self, encoded: list[tuple[list[int], list[int]]]):
        self.data = encoded

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self.data[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def _make_collate_fn(pad_id: int = 0):
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        inputs, targets = zip(*batch)
        max_len = max(x.size(0) for x in inputs)
        padded = torch.zeros(len(inputs), max_len, dtype=torch.long) + pad_id
        for i, x in enumerate(inputs):
            padded[i, : x.size(0)] = x
        targets_stacked = torch.stack(targets, dim=0)
        return padded, targets_stacked

    return collate_fn


class DatasetBuilder:
    def __init__(
        self,
        sentences: list[str],
        final_seed: int,
        predict_n: int,
        data_cfg: dict,
        batch_size: int,
    ):
        self.sentences = sentences
        self.final_seed = final_seed
        self.predict_n = predict_n
        self.train_ratio = data_cfg["train_ratio"]
        self.test_ratio = data_cfg["test_ratio"]
        self.sentence_min_len = data_cfg["sentence_min_len"]
        self.batch_size = batch_size

        assert self.sentence_min_len - self.predict_n >= 1, (
            f"sentence_min_len ({self.sentence_min_len}) - predict_n ({self.predict_n}) "
            f"must be >= 1 to have at least one input token."
        )

    def _split(self, sentences: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(sentences)
        train_end = int(n * self.train_ratio)
        test_end = train_end + int(n * self.test_ratio)
        return sentences[:train_end], sentences[test_end:], sentences[train_end:test_end]

    def _build_vocab(self, train_sentences: list[str]) -> dict[str, int]:
        tokens = set()
        for sentence in train_sentences:
            for word in sentence.split():
                tokens.add(word)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word in sorted(tokens):
            vocab[word] = len(vocab)
        return vocab

    def _encode(
        self, sentences: list[str], vocab: dict[str, int]
    ) -> list[tuple[list[int], list[int]]]:
        unk_id = vocab["<UNK>"]
        encoded = []
        for sentence in sentences:
            tokens = sentence.split()
            ids = [vocab.get(t, unk_id) for t in tokens]
            k = self.predict_n
            input_ids = ids[: len(ids) - k]
            target_ids = ids[len(ids) - k :]
            encoded.append((input_ids, target_ids))
        return encoded

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
        random.seed(self.final_seed)
        shuffled = self.sentences[:]
        random.shuffle(shuffled)

        train_sents, eval_sents, test_sents = self._split(shuffled)
        vocab = self._build_vocab(train_sents)

        train_enc = self._encode(train_sents, vocab)
        eval_enc = self._encode(eval_sents, vocab)
        test_enc = self._encode(test_sents, vocab)

        collate_fn = _make_collate_fn(pad_id=0)

        train_loader = DataLoader(
            _SentenceDataset(train_enc),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        eval_loader = DataLoader(
            _SentenceDataset(eval_enc),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            _SentenceDataset(test_enc),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return train_loader, eval_loader, test_loader, vocab
