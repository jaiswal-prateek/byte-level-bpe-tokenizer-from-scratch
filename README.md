# Byte-Level BPE Tokenizer from Scratch

A hands-on project to understand how byte level BPE tokenization works by implementing it from scratch in Python and experimenting with vocabulary sizes, encoding/decoding, and token efficiency.

I started with basic character and word tokenization, then moved to implementing byte level BPE to understand how a tokenizer learns its vocabulary, applies merge rules during encoding, and converts token IDs back into the original text.

This is a learning project focused on understanding the implementation through hands-on experiments, rather than building a production ready tokenizer.

## What is implemented

- Basic character, word, and BPE experiments in notebooks.
- Byte-level BPE tokenizer implemented from scratch.
- Pre-tokenization, pair counting, vocabulary building, and merge rules.
- Encoding text into token IDs and decoding it back.
- Saving and loading the trained tokenizer.
- Training scripts to experiment with different vocabulary sizes.
- Evaluation across English, technical text, code, SQL, URLs, numbers, Hindi, Japanese, Chinese, and other symbols.
- Experiments comparing tokenization efficiency across vocabulary sizes.

## Current Status and Next Steps

This is an educational implementation, and I am improving it incrementally as I learn more about tokenization and BPE.

### Next Steps

- Fix and rerun category-level evaluation.
- Add automated tests for encoding/decoding and save/load.
- Improve training efficiency by reducing repeated pair counting and corpus scans.
- Explore special-token handling and improve pre-tokenization.
- Evaluate on larger held-out datasets and compare results across domains.

These are planned improvements. I will work through them one at a time as I continue experimenting with the implementation.

## Repository structure

```text
byte-level-bpe-tokenizer-from-scratch/
├── artifacts/              # Saved tokenizer vocabularies and merge rules
├── data/
│   ├── eval/                # Evaluation text
│   └── raw/                 # Downloaded training data (not tracked)
├── docs/                    # Concepts, implementation notes and findings
├── notebooks/               # Hands-on experiments
├── scripts/
│   ├── download_tinystories.py
│   ├── train_tokenizer.py
│   └── download_fineweb.py  # Dataset download script, if present
├── tokenizer/
│   └── bpe.py               # Main tokenizer implementation
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/jaiswal-prateek/byte-level-bpe-tokenizer-from-scratch.git
cd byte-level-bpe-tokenizer-from-scratch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The download script uses Hugging Face Datasets to download TinyStories. Internet access is required when downloading the dataset.

## Train the tokenizer

I used the `karpathy/tinystories-gpt4-clean` dataset and downloaded 100,000 stories for the initial experiments.

### 1. Download TinyStories

```bash
python scripts/download_tinystories.py
```

This creates the training file under:

```text
data/raw/tinystories.jsonl
```

The raw dataset is not included in the repository, so download it before training a new tokenizer.

### 2. Train with an 8K vocabulary

```bash
python scripts/train_tokenizer.py --vocab-size 8192
```

By default, the tokenizer is saved to:

```text
artifacts/tokenizer_8192.json
```

### 3. Train with a different vocabulary size

For 16K:

```bash
python scripts/train_tokenizer.py \
    --vocab-size 16384 \
    --output artifacts/tokenizer_16384.json
```

For 32K:

```bash
python scripts/train_tokenizer.py \
    --vocab-size 32768 \
    --output artifacts/tokenizer_32768.json
```

The requested vocabulary size is a target. The actual size can be smaller if the training process runs out of mergeable pairs.

You can also limit the number of stories used for a quick experiment:

```bash
python scripts/train_tokenizer.py \
    --vocab-size 8192 \
    --max-stories 1000 \
    --output artifacts/tokenizer_test.json
```

## Use a trained tokenizer

The saved JSON files contain the vocabulary and merge information needed to load a trained tokenizer.

For example:

```python
from tokenizer.bpe import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.load("artifacts/tokenizer_8192.json")

text = "Artificial intelligence is interesting."

tokens = tokenizer.encode(text)
decoded_text = tokenizer.decode(tokens)

print("Token IDs:", tokens)
print("Decoded text:", decoded_text)

assert decoded_text == text
```

The assertion checks that the original text is recovered after encoding and decoding.

You can load the 16K or 27.6K tokenizer in the same way by changing the file path.

## Evaluate tokenization efficiency

The repository contains a multilingual evaluation corpus at:

```text
data/eval/multi_language_text.txt
```

It includes different types of text so I can compare how the trained tokenizers behave beyond the TinyStories training data.

The existing tokenizer artifacts can be used directly for evaluation. You do not need to download TinyStories or retrain the tokenizer just to run the evaluation notebook.

Open the notebooks in Jupyter or VS Code and run the cells in order:

1. `notebooks/01_tokenization_fundamentals.ipynb` — basic tokenization and BPE experiments.
2. `notebooks/02_tokenizer_evaluation.ipynb` — comparison of trained tokenizers on evaluation text.

The evaluation compares vocabulary size, total token count, tokens per character, characters per token, and token reduction relative to the 8K tokenizer.

## Initial results

I trained three tokenizers on 100,000 TinyStories and evaluated them on the same 11,684-character multilingual corpus.

| Vocabulary | Actual size | Tokens produced | Reduction vs 8K |
|---|---:|---:|---:|
| 8K | 8,192 | 6,821 | — |
| 16K | 16,384 | 6,356 | 6.82% |
| 32K target | 27,638 | 5,878 | 13.82% |

The 32K run stopped at 27,638 tokens because no more mergeable pairs were available under the current implementation and training corpus.

The larger vocabularies produced fewer tokens on this evaluation corpus. The amount of improvement varied by category, so vocabulary size alone does not tell the complete story.

These are initial results from my implementation and evaluation data, not a benchmark against production tokenizers. The category-level evaluation also needs a correction to its section-heading parsing before those results can be treated as final.

More details and experiment notes are available in the `docs/` directory.

## Documentation

- [Tokenization Fundamentals](docs/01_tokenization_fundamentals.md) — basic concepts and initial experiments.
- [Byte-Level BPE Implementation](docs/02_byte_level_bpe.md) — implementation details and what I learned while building it.
- [Experiment Log](docs/03_experiment_log.md) — training results, evaluation findings, and current limitations.

## Current limitations

This is an educational implementation, and there are a few things I still need to improve:

- The pre-tokenization regex is simplified.
- Training becomes slower as vocabulary size increases because pair counts and the corpus are repeatedly scanned.
- Special-token handling needs further work.
- The evaluation corpus is small and manually constructed.
- Tokenization efficiency does not tell us by itself how well a language model will train or generate text with a particular tokenizer.

I am working through these parts incrementally to understand the behavior and trade-offs before making the implementation more complex.

## Why this project?

I am learning how language models work from the fundamentals, and tokenization is one of the components I wanted to understand properly before moving further into model training.

The goal is to build a clear understanding through implementation and experiments, not just use existing libraries and APIs.