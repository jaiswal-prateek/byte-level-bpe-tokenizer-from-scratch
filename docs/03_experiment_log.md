# Tokenizer Experiments and Findings

This document captures the experiments I ran while building the byte-level BPE tokenizer, along with the results and observations from those experiments.

The goal was to understand how the tokenizer behaves in practice, rather than only understanding the algorithm theoretically.

## 1. Experiments with basic tokenization

**Notebook:** [`01_tokenization_fundamentals.ipynb`](../notebooks/01_tokenization_fundamentals.ipynb)

I started with character-level and word-level tokenization before implementing a basic BPE tokenizer.

I experimented with:
- Character and word tokenization.
- Handling words that are not present in the vocabulary.
- Why subword tokenization is useful.
- How BPE selects and merges frequent adjacent pairs.
- How the number of merges affects vocabulary size and token count.
- Encoding text into token IDs and decoding it back.

My main understanding from these experiments was that BPE repeatedly merges frequent adjacent pairs in the current tokenized corpus. After every merge, pair frequencies need to be recalculated because the corpus representation has changed.

I also observed that increasing the number of merges generally increases vocabulary size and reduces the number of tokens needed to represent the same text.

## 2. Byte-level BPE training

**Implementation:** [`bpe.py`](../tokenizer/bpe.py)

**Training script:** [`train_tokenizer.py`](../scripts/train_tokenizer.py)

I implemented a reusable byte-level BPE tokenizer and trained it on 100,000 TinyStories from the `karpathy/tinystories-gpt4-clean` dataset.

I trained tokenizers with target vocabulary sizes of 8K, 16K, and 32K.

### Training results

| Target vocabulary | Actual vocabulary | Learned merges | Training time |
|---|---:|---:|---:|
| 8,192 | 8,192 | 7,936 | 84.67 sec |
| 16,384 | 16,384 | 16,128 | 146.56 sec |
| 32,768 | 27,638 | 27,382 | 207.57 sec |

The 32K run stopped at 27,638 tokens because no more mergeable pairs were available under the current implementation and training corpus.

These are the results from my local runs, not general performance benchmarks.

One implementation detail I learned was that storing unique pre-tokenized pieces with their frequencies allows pair counts to be weighted by how often each piece occurs. This avoids storing every repeated piece separately.

## 3. Comparing vocabulary sizes on evaluation text

**Notebook:** [`02_tokenizer_evaluation.ipynb`](../notebooks/02_tokenizer_evaluation.ipynb)

**Evaluation corpus:** [`multi_language_text.txt`](../data/eval/multi_language_text.txt)

I wanted to compare the trained tokenizers on the same text, rather than only looking at vocabulary sizes.

The evaluation corpus contained 11,684 characters and included English prose, technical text, code, SQL, URLs, numbers, multilingual text, and other symbols.

I measured the number of tokens produced by each tokenizer and calculated tokens per character and characters per token.

### Results

| Tokenizer | Vocabulary size | Characters | Tokens | Tokens/character | Token reduction vs 8K |
|---|---:|---:|---:|---:|---:|
| 8K | 8,192 | 11,684 | 6,821 | 0.5838 | — |
| 16K | 16,384 | 11,684 | 6,356 | 0.5440 | 6.82% |
| 27.6K | 27,638 | 11,684 | 5,878 | 0.5031 | 13.82% |

### What I observed

The larger vocabularies produced fewer tokens for the same evaluation text.

The 16K tokenizer produced approximately 6.82% fewer tokens than the 8K tokenizer. The 27.6K tokenizer produced approximately 13.82% fewer tokens than the 8K tokenizer.

Increasing vocabulary size therefore improved tokenization efficiency on this particular corpus, although the reduction in token count was not proportional to the increase in vocabulary size.

These results measure tokenization efficiency only. I have not yet compared how these tokenizers affect language-model training or generation.

## 4. Comparing tokenization across categories

I also split the evaluation corpus into categories and compared tokens per character across the three tokenizers.

The initial results showed different levels of improvement across categories.

For example:

| Category | 8K tokens | 16K tokens | 27.6K tokens |
|---|---:|---:|---:|
| English prose | 320 | 269 | 234 |
| Technical text | 369 | 328 | 300 |

For English prose, the token count decreased from 320 with the 8K tokenizer to 234 with the 27.6K tokenizer.

For technical text, it decreased from 369 to 300.

I also observed that some categories, such as Hindi, showed much smaller improvements in the initial results than English prose.

These observations are preliminary. I identified an issue in the section-splitting logic: it treated lines beginning with `# ` as category headings, including some comments inside code examples. The category-level results need to be rerun after fixing the headings before I treat them as final.

## 5. Current understanding

From the experiments completed so far, I have learned the following:

- BPE learns merge rules from adjacent-pair frequencies in the training corpus.
- Byte-level BPE starts with 256 possible byte tokens and builds larger tokens through merges.
- Training and encoding are different processes: training learns merge rules, while encoding applies those rules in rank order.
- A larger vocabulary reduced token count in my evaluation experiments.
- The amount of improvement varied across the text categories I tested.
- Tokenization efficiency alone does not establish which tokenizer is best for training a language model.

## 6. Work not completed yet

The following are possible next steps, not completed experiments:

- Fix the category-heading issue and rerun the category comparison.
- Inspect token boundaries for representative English, Python, SQL, and multilingual examples.
- Add automated tests for encoding, decoding, and saving/loading.
- Evaluate on larger held-out datasets.
- Investigate ways to improve BPE training performance.
- Compare the tokenizers in controlled language-model training experiments.

I will work through these incrementally rather than changing several parts of the tokenizer at once.