from pathlib import Path
import json
from datasets import load_dataset


DATASET_NAME = "karpathy/tinystories-gpt4-clean"

START_INDEX = 0
NUM_STORIES = 100_000

RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "tinystories.jsonl"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    end_index = min(
        START_INDEX + NUM_STORIES,
        len(dataset),
    )

    selected_data = dataset.select(
        range(START_INDEX, end_index)
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for row in selected_data:
            file.write(
                json.dumps(
                    {"text": row["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved stories: {len(selected_data):,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()