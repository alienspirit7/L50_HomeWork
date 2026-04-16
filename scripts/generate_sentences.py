"""
Generate 100,000 natural English sentences (5-6 words each) using Ollama.
Writes to input/sentences.txt and logs vocab stats to output/logs/vocab_stats.txt.
Saves progress every 1,000 sentences to output/logs/sentences_checkpoint.txt.

Usage:
    python scripts/generate_sentences.py
    python scripts/generate_sentences.py --resume
    python scripts/generate_sentences.py --target 100000 --batch 50 --model llama3.2:3b
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_DEFAULT = 100_000
BATCH_DEFAULT = 50
MODEL_DEFAULT = "llama3.2:3b"
CHECKPOINT_EVERY = 1_000

CHECKPOINT_PATH = PROJECT_ROOT / "output" / "logs" / "sentences_checkpoint.txt"
OUT_PATH = PROJECT_ROOT / "input" / "sentences.txt"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "vocab_stats.txt"

PROMPT_TEMPLATE = """Generate exactly {n} natural English sentences. Rules:
- Each sentence must be exactly 5 or 6 words long
- Use everyday vocabulary, varied topics
- No numbering, no bullet points, no punctuation except apostrophes
- One sentence per line, nothing else

Output only the sentences, one per line."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sentences via Ollama.")
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if it exists")
    return parser.parse_args()


def is_valid(sentence: str) -> bool:
    words = sentence.strip().split()
    if len(words) < 5 or len(words) > 6:
        return False
    if re.search(r'[0-9]', sentence):
        return False
    return True


def clean(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r'^[\d\.\-\*\•]+\s*', '', sentence)
    sentence = re.sub(r'[^\w\s\']', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence.lower()


def generate_batch(model: str, n: int) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(n=n)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        raw = response.get("response", "")
    except Exception as e:
        print(f"  [warn] Ollama error: {e}")
        return []
    results = []
    for line in raw.strip().splitlines():
        cleaned = clean(line)
        if cleaned and is_valid(cleaned):
            results.append(cleaned)
    return results


def load_checkpoint() -> tuple[list[str], set[str]]:
    if not CHECKPOINT_PATH.exists():
        return [], set()
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"[resume] loaded {len(sentences):,} sentences from checkpoint")
    return sentences, set(sentences)


def save_checkpoint(sentences: list[str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")


def write_vocab_stats(sentences: list[str]) -> None:
    word_counts: Counter = Counter()
    for s in sentences:
        word_counts.update(s.split())
    unique_words = len(word_counts)
    total_words = sum(word_counts.values())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"total_sentences: {len(sentences)}\n")
        f.write(f"total_words: {total_words}\n")
        f.write(f"unique_words: {unique_words}\n")
        f.write(f"avg_words_per_sentence: {total_words / max(len(sentences), 1):.2f}\n")
        f.write("\ntop_50_words:\n")
        for word, count in word_counts.most_common(50):
            f.write(f"  {word}: {count}\n")
    print(f"[vocab] unique words: {unique_words:,} across {len(sentences):,} sentences")


def main() -> None:
    args = parse_args()

    if args.resume and CHECKPOINT_PATH.exists():
        sentences, seen = load_checkpoint()
    else:
        sentences, seen = [], set()

    print(f"[config] model={args.model}  target={args.target:,}  batch={args.batch}")
    print(f"[start] {len(sentences):,} sentences already collected")

    last_checkpoint = len(sentences)

    while len(sentences) < args.target:
        batch = generate_batch(args.model, args.batch)
        for s in batch:
            if s not in seen:
                seen.add(s)
                sentences.append(s)
                if len(sentences) >= args.target:
                    break

        if len(sentences) - last_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(sentences)
            last_checkpoint = len(sentences)
            print(f"  [progress] {len(sentences):,} / {args.target:,}  [checkpoint saved]")

    print(f"[done] {len(sentences):,} unique valid sentences")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")
    print(f"[saved] {OUT_PATH}")

    if CHECKPOINT_PATH.exists():
        os.remove(CHECKPOINT_PATH)
        print(f"[cleanup] checkpoint removed")

    write_vocab_stats(sentences)


if __name__ == "__main__":
    main()
