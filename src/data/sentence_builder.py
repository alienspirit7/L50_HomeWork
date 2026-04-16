import os
import random
from datetime import datetime


class SentenceBuilder:
    def __init__(self, data_cfg: dict, project_root: str):
        self.project_root = project_root
        self.sentences_path = os.path.join(project_root, data_cfg["sentences_path"]) if data_cfg.get("sentences_path") else None
        self.words_path = os.path.join(project_root, data_cfg["words_path"])
        self.num_sentences = data_cfg["num_sentences"]
        self.sentence_min_len = data_cfg["sentence_min_len"]
        self.sentence_max_len = data_cfg["sentence_max_len"]
        self.dedup_threshold = data_cfg["dedup_threshold"]
        self.max_dedup_retries = data_cfg["max_dedup_retries"]
        self.base_seed = data_cfg["data_seed"]
        self.log_path = os.path.join(project_root, "output", "logs", "dataset_stats.txt")

    def _load_vocab(self) -> list[str]:
        with open(self.words_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _generate(self, vocab: list[str], seed: int) -> list[str]:
        random.seed(seed)
        sentences = []
        for _ in range(self.num_sentences):
            length = random.randint(self.sentence_min_len, self.sentence_max_len)
            words = random.sample(vocab, length)
            sentences.append(" ".join(words))
        return sentences

    def _dedup(self, sentences: list[str]) -> list[str]:
        seen = set()
        result = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def _log(self, attempt: int, seed: int, dup_rate: float, status: str):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"[{timestamp}] attempt={attempt} seed={seed} "
            f"dup_rate={dup_rate:.6f} status={status}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _load_sentences(self) -> list[str]:
        with open(self.sentences_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def build(self) -> tuple[list[str], int]:
        # If a pre-generated sentences file exists, use it directly
        if self.sentences_path and os.path.exists(self.sentences_path):
            sentences = self._load_sentences()
            deduped = self._dedup(sentences)
            total = len(sentences)
            dup_rate = (total - len(deduped)) / max(total, 1)
            self._log(1, self.base_seed, dup_rate, "PASS-FILE")
            return deduped, self.base_seed

        # Otherwise generate from words.txt
        vocab = self._load_vocab()
        total = self.num_sentences

        for attempt in range(1, self.max_dedup_retries + 1):
            seed = self.base_seed + (attempt - 1)
            sentences = self._generate(vocab, seed)
            unique_count = len(set(sentences))
            dup_rate = (total - unique_count) / total

            if dup_rate < self.dedup_threshold:
                self._log(attempt, seed, dup_rate, "PASS")
                return self._dedup(sentences), seed
            elif attempt < self.max_dedup_retries:
                self._log(attempt, seed, dup_rate, "RETRY")
            else:
                self._log(attempt, seed, dup_rate, "FAIL")
                raise RuntimeError(
                    f"Dedup threshold not met after {self.max_dedup_retries} attempts. "
                    f"Final: attempt={attempt} seed={seed} dup_rate={dup_rate:.6f} "
                    f"threshold={self.dedup_threshold}"
                )

        raise RuntimeError("Unexpected exit from dedup retry loop.")
