import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.bpe import ByteLevelBPETokenizer


def load_texts(path: Path, max_stories: int | None = None) -> list[str]:
    texts = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if max_stories is not None and len(texts) >= max_stories:
                break

            record = json.loads(line)
            texts.append(record["text"])

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/tinystories.jsonl"),
        help="Path to training JSONL file.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Target tokenizer vocabulary size.",
    )

    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Maximum number of stories to use.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tokenizer_8192.json"),
        help="Path to save the trained tokenizer.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Training data not found: {args.input}"
        )

    if args.vocab_size <= 256:
        raise ValueError(
            "vocab-size must be greater than 256."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading data: {args.input}")

    start_time = time.perf_counter()

    texts = load_texts(
        args.input,
        max_stories=args.max_stories,
    )

    print(f"Stories loaded: {len(texts):,}")
    print(f"Vocabulary size: {args.vocab_size:,}")

    tokenizer = ByteLevelBPETokenizer()

    print("Training tokenizer...")

    tokenizer.train(
        texts=texts,
        vocab_size=args.vocab_size,
    )

    tokenizer.save(args.output)

    elapsed = time.perf_counter() - start_time

    print("\nTraining complete.")
    print(f"Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"Merges learned: {len(tokenizer.merges):,}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Saved tokenizer: {args.output}")


if __name__ == "__main__":
    main()