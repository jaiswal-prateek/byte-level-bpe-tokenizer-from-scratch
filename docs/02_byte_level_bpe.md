# Byte-Level BPE Tokenizer — Learning and Implementation

## 1. Why I started this

After building my Mini-GPT from scratch, I wanted to understand tokenization more deeply rather than treating the tokenizer as a black box.

I started with a simple character tokenizer and then implemented a basic word tokenizer. From there, I explored why subword tokenization is useful and how Byte Pair Encoding (BPE) builds a vocabulary by merging frequently occurring adjacent tokens.

The goal was to understand the complete process:

- How raw text gets converted into tokens.
- How BPE learns its vocabulary.
- How the learned merges are used during encoding.
- How token IDs can be converted back into the original text.
- How vocabulary size affects tokenization efficiency.

I first implemented a small, character-based BPE in a notebook. Once I understood the basic idea, I started building a reusable byte-level BPE tokenizer in Python.

## 2. What I learned before implementing byte-level BPE

### Character tokenization

In character tokenization, each character becomes a separate token.

For example:

```text
hello
→ h e l l o
```

This is easy to implement and understand, but sequences become long because every character needs its own token.

### Word tokenization

A word tokenizer treats each word as a token.

```text
hello world
→ hello | world
```

This produces shorter sequences, but it has another problem: what happens when the tokenizer encounters a word that was not present in its vocabulary?

One possible solution is an unknown token such as `<unk>`, but that loses the identity of the original word.

### Subword tokenization

Subword tokenization tries to find a middle ground between character-level and word-level tokenization.

Common words or word fragments can become individual tokens, while less common words can be represented using smaller pieces.

For example, a word might be represented as:

```text
unhappiness
→ un | happiness
```

The exact segmentation depends on the tokenizer and its learned vocabulary.

### How BPE works

My understanding of BPE is that it repeatedly performs the following steps:

1. Start with a basic vocabulary.
2. Count adjacent token pairs in the current corpus.
3. Select the most frequent pair.
4. Merge that pair into a new token.
5. Update the tokenized corpus.
6. Count pairs again and repeat.

The important part is that pair frequencies are recalculated after each merge. The next pair is selected based on the updated corpus, not the original frequencies.

I experimented with different numbers of merges in the notebook and observed that more merges generally produce a larger vocabulary and fewer tokens.

## 3. Why byte-level BPE?

My initial character-based BPE depended on the characters present in its starting vocabulary. If it encountered an unseen character, it could not represent it unless additional handling was implemented.

Byte-level BPE addresses this by starting with all 256 possible byte values.

The initial vocabulary is:

```python
self.vocab = {i: bytes([i]) for i in range(256)}
```

Each token ID from `0` to `255` represents one byte.

For example:

```python
bytes([56])  # b'8'
bytes([65])  # b'A'
```

These are byte values, not character-to-ID mappings in the usual sense.

BPE then learns additional tokens by merging pairs of existing tokens. These new tokens represent longer byte sequences.

For example, the vocabulary could conceptually contain:

```text
65  → b'A'
112 → b'p'
256 → b'ap'
257 → b'app'
```

The actual learned token IDs depend on the training corpus and merge order.

This approach gives the tokenizer a way to represent arbitrary text through its UTF-8 bytes, including text containing characters from different writing systems.

It does not mean every language will be tokenized efficiently. Representability and tokenization efficiency are two different things.

## 4. Repository structure

I separated the tokenizer implementation, training scripts, datasets, notebooks, documentation, and saved artifacts.

```text
byte-level-bpe-tokenizer-from-scratch/
├── artifacts/
│   ├── tokenizer_8192.json
│   ├── tokenizer_16384.json
│   └── tokenizer_32768.json
├── data/
│   ├── eval/
│   │   ├── general_text.txt
│   │   └── multi_language_text.txt
│   └── raw/
│       └── tinystories.jsonl
├── docs/
│   ├── 01_tokenization_fundamentals.md
│   └── 02_byte_level_bpe.md
├── notebooks/
│   ├── 01_tokenization_fundamentals.ipynb
│   └── 02_tokenizer_evaluation.ipynb
├── scripts/
│   ├── download_tinystories.py
│   └── train_tokenizer.py
├── tokenizer/
│   └── bpe.py
├── README.md
└── requirements.txt
```

The purpose of this structure is to keep experiments reproducible and separate from the reusable tokenizer implementation.

## 5. Implementing the tokenizer

I implemented the tokenizer in `tokenizer/bpe.py`.

The main class is:

```python
class ByteLevelBPETokenizer:
    ...
```

The implementation covers pre-tokenization, vocabulary initialization, pair counting, merging, training, encoding, decoding, and saving/loading the tokenizer.

### 5.1 Pre-tokenization

Before applying BPE, I split the input text into smaller pieces using a regular expression.

```python
def pretokenize(self, text: str) -> list[str]:
    pattern = r" ?[A-Za-z]+(?:'[A-Za-z]+)?| ?[0-9]+|[^\w\s]+|\s+"
    return re.findall(pattern, text)
```

The idea is to split text into pieces such as words, numbers, punctuation, and whitespace before applying BPE to the bytes of each piece.

This is an important design decision: BPE does not operate over the entire raw text as one unrestricted sequence in this implementation. It operates on the UTF-8 bytes of each pre-tokenized piece.

The current regular expression is also a simplification. It is not a complete implementation of the pre-tokenization rules used by production tokenizers.

### 5.2 Building the piece corpus

I initially considered counting pairs across every occurrence of every piece in the dataset.

Instead, I store each unique pre-tokenized piece once and separately track how frequently it occurs.

```python
def build_piece_corpus(self, texts: list[str]):
    piece_counts = Counter()

    for text in texts:
        pieces = self.pretokenize(text)
        piece_counts.update(pieces)

    piece_corpus = {
        piece: list(piece.encode("utf-8"))
        for piece in piece_counts
    }

    return piece_corpus, piece_counts
```

For example, if the piece `hello` appears 1,000 times, I store its byte sequence once and record its frequency as 1,000.

This avoids storing and repeatedly processing identical pieces as separate entries.

### 5.3 Counting pairs using piece frequency

Since each unique piece is stored only once, its pair counts need to be weighted by how often that piece occurs.

```python
def count_pairs_weighted(self, piece_corpus, piece_counts):
    pair_counts = {}

    for piece, tokens in piece_corpus.items():
        frequency = piece_counts[piece]

        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = (
                pair_counts.get(pair, 0) + frequency
            )

    return pair_counts
```

The key insight here is that the frequency of a pair across the corpus is not the same as its frequency within one unique piece.

If a pair occurs once in a piece that appears 1,000 times, it contributes 1,000 to the corpus-level count.

### 5.4 Merging a selected pair

I implemented a separate method to merge a pair of adjacent token IDs.

```python
def merge_pair(self, tokens, pair, new_token_id):
    merged = []
    i = 0

    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and (tokens[i], tokens[i + 1]) == pair
        ):
            merged.append(new_token_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged
```

This method does not decide which pair should be merged. It only replaces occurrences of the pair passed to it.

For example:

```text
Before: A p p l e
Merge:  (p, p)

After:  A pp l e
```

The same pair is merged from left to right, and a token consumed by a merge is not reused in another merge during that same pass.

Keeping pair selection and pair merging separate made the implementation easier to understand.

### 5.5 Training BPE

The training process connects the methods above.

The high-level sequence is:

```text
Raw texts
    ↓
Pre-tokenize
    ↓
Convert pieces to UTF-8 bytes
    ↓
Initialize 256 byte tokens
    ↓
Count frequency-weighted adjacent pairs
    ↓
Select the most frequent pair
    ↓
Create a new token
    ↓
Update the piece corpus
    ↓
Repeat
```

I use `vocab_size` as the training argument rather than asking for a number of merges directly.

Since the initial vocabulary contains 256 byte tokens, the maximum number of new merge tokens is:

```python
max_merges = vocab_size - 256 - len(self.special_tokens)
```

Special tokens are not currently part of my main experiments, but the calculation allows for them in the vocabulary-size accounting.

At each iteration, the implementation selects the most frequent pair:

```python
best_pair = max(pair_counts, key=pair_counts.get)
```

It assigns a new token ID, stores the merge rule and its rank, constructs the new token's bytes by concatenating the bytes of the two existing tokens, and updates the corpus.

The merge information is stored in two mappings:

- `merges`: pair of token IDs → resulting token ID.
- `merge_ranks`: pair of token IDs → rank at which the merge was learned.

The vocabulary stores the resulting byte sequence for every token ID.

These mappings have different purposes and are all required by the current implementation.

### 5.6 Encoding text

Training learns the vocabulary and merge rules. Encoding applies those learned rules to new text.

The process is:

```text
Input text
    ↓
Pre-tokenize
    ↓
Convert each piece to UTF-8 bytes
    ↓
Find applicable adjacent pairs
    ↓
Select the pair with the lowest learned merge rank
    ↓
Merge and repeat
    ↓
Return token IDs
```

For one piece, I first convert the UTF-8 bytes into their initial token IDs:

```python
tokens = list(piece.encode("utf-8"))
```

Then I repeatedly identify adjacent pairs that exist in `merge_ranks`.

The pair with the lowest rank is selected because it was learned earliest during training:

```python
_, best_pair = min(pair_candidates)
```

After merging, the process repeats until no applicable learned pair remains.

This is an important distinction: encoding does not choose the currently most frequent pair. It follows the learned merge order.

The complete `encode()` method applies this process to each pre-tokenized piece and concatenates the resulting token IDs.

### 5.7 Decoding text

Decoding reverses the representation process.

Each token ID is mapped back to its byte sequence using the vocabulary. The byte sequences are concatenated, and the resulting byte stream is decoded as UTF-8.

Conceptually:

```text
Token IDs
    ↓
Look up each token's bytes
    ↓
Concatenate bytes
    ↓
Decode UTF-8
    ↓
Original text
```

I concatenate the bytes before decoding because an individual token may contain only part of a multibyte UTF-8 character.

The main correctness check is:

```python
text = "Hello, नमस्ते, こんにちは, 你好!"

tokens = tokenizer.encode(text)
decoded_text = tokenizer.decode(tokens)

assert decoded_text == text
```

The exact token IDs are not important for this test. The important property is that decoding reproduces the original input exactly.

## 6. Training on TinyStories

I used the `karpathy/tinystories-gpt4-clean` dataset through the Hugging Face `datasets` library.

I downloaded 100,000 stories into:

```text
data/raw/tinystories.jsonl
```

Each line contains a JSON record with the story text.

I created `scripts/train_tokenizer.py` to load this file, train the tokenizer, and save the resulting vocabulary and merge information.

The script accepts arguments for the input file, vocabulary size, maximum number of stories, and output path.

For example:

```bash
python scripts/train_tokenizer.py --vocab-size 8192
```

```bash
python scripts/train_tokenizer.py \
    --vocab-size 16384 \
    --output artifacts/tokenizer_16384.json
```

```bash
python scripts/train_tokenizer.py \
    --vocab-size 32768 \
    --output artifacts/tokenizer_32768.json
```

This allows me to repeat the training process with different vocabulary sizes without modifying the implementation.

## 7. Vocabulary-size experiments

I trained three tokenizers on the same 100,000-story dataset.

| Target vocabulary | Actual vocabulary | Learned merges | Training time |
|---|---:|---:|---:|
| 8K | 8,192 | 7,936 | 84.67 seconds |
| 16K | 16,384 | 16,128 | 146.56 seconds |
| 32K | 27,638 | 27,382 | 207.57 seconds |

The 32K experiment did not reach the requested vocabulary size. Pair counting eventually returned no more mergeable pairs under the current pre-tokenization and training corpus, so training stopped early.

These measurements are specific to my implementation, machine, dataset, and experimental setup. They should not be treated as universal training-time benchmarks.

One limitation I noticed is that the training loop repeatedly scans the unique-piece corpus after every merge. This is straightforward to understand but becomes a bottleneck as the vocabulary and corpus grow.

I decided to keep the implementation simple for now and postpone optimization until I understand and validate the complete tokenizer pipeline.

## 8. Evaluating tokenization efficiency

I wanted to understand whether a larger vocabulary actually improves tokenization and whether the improvement holds outside the training corpus.

I therefore created two evaluation files:

```text
data/eval/general_text.txt
data/eval/multi_language_text.txt
```

The multilingual evaluation corpus includes English prose, technical text, Python, SQL, URLs, file paths, numbers, long words and identifiers, Hindi, Japanese, Chinese, emojis, and mixed Hindi-English text.

I used the same evaluation text for all three tokenizers.

The main metrics were:

- **Tokens:** Number of token IDs produced.
- **Tokens per character:** Token count divided by the number of characters.
- **Characters per token:** Character count divided by the number of tokens.
- **Token reduction:** Percentage fewer tokens relative to the 8K tokenizer.

The overall results for the 11,684-character evaluation corpus were:

| Tokenizer | Vocabulary | Tokens | Tokens/character | Token reduction vs 8K |
|---|---:|---:|---:|---:|
| 8K | 8,192 | 6,821 | 0.584 | — |
| 16K | 16,384 | 6,356 | 0.544 | 6.82% |
| 27.6K | 27,638 | 5,878 | 0.503 | 13.82% |

Increasing the vocabulary reduced the number of tokens needed to represent the same text.

However, the reduction was not proportional to the increase in vocabulary size. Increasing the vocabulary from 8K to approximately 27.6K reduced the token count by about 13.8%, rather than anything close to 3.4 times.

This was my first practical demonstration of diminishing returns from increasing vocabulary size.

### Comparing categories

I also broke the evaluation corpus into sections and calculated tokenization efficiency separately for each category.

The initial results suggested that English prose benefited considerably from the larger vocabularies, while Hindi showed very little improvement. Python and SQL also showed larger gains at the 27.6K vocabulary size than at 16K.

This makes sense as an experiment: the tokenizer learned its merges from TinyStories, so its learned patterns reflect the training corpus. A larger vocabulary does not guarantee equally good compression across every language or domain.

I also found a problem in my evaluation code: splitting on every line beginning with `# ` caused Python comments to be treated as section headings. I need to use a dedicated heading format, such as `## Section Name`, and rerun the category-level comparison before treating those results as final.

The evaluation corpus is useful for exploratory comparisons, but it is still small and manually constructed. It is not enough to establish general performance across all languages or real-world workloads.

## 9. Current limitations

This project is an educational implementation, not a production tokenizer.

The main limitations I have identified are:

1. **Simplified pre-tokenization:** My regular expression is not equivalent to the pre-tokenization rules used by production tokenizers.
2. **Training performance:** The implementation recalculates pair frequencies and scans the corpus for every merge. This is easy to follow but inefficient at larger scales.
3. **Special tokens:** The current experiments focus on ordinary text. Special-token handling and its interaction with encoding need further work.
4. **Limited evaluation data:** The multilingual corpus is useful for exploratory testing, but larger held-out datasets are needed for more reliable comparisons.
5. **Domain-specific compression:** The tokenizer was trained on TinyStories, so it may not perform equally well on code, technical documents, or other languages.
6. **Token count is not the whole story:** A larger vocabulary can reduce sequence length while increasing the number of parameters required by a language model's embedding and output layers.

These limitations give me a clear direction for future experiments without needing to solve everything at once.

## 10. What I learned

The most useful part of this project was implementing the algorithm rather than only reading about it.

I now have a clearer understanding of the difference between training and encoding. Training selects frequent pairs and records merge rules. Encoding applies the learned rules in rank order. Decoding uses the vocabulary to reconstruct the original bytes and then the original text.

I also learned why the vocabulary, merge rules, and merge ranks need to be stored separately, why pair frequencies must account for repeated pieces, and why byte-level representation helps with text containing unseen characters.

The experiments showed me that increasing vocabulary size can improve compression, but the gains depend on the corpus and eventually diminish. The category-level evaluation also highlighted the importance of checking how a tokenizer behaves outside its training distribution.

For now, I am keeping the implementation understandable and reproducible. I will revisit performance optimization, more rigorous evaluation, and special-token handling as separate steps rather than introducing too many changes at once.

## 11. Next steps

The next stage will be incremental:

- Fix and rerun the category-level evaluation.
- Inspect actual token boundaries for representative English, Python, SQL, and multilingual examples.
- Add automated tests for encoding, decoding, and save/load behavior.
- Review special-token handling.
- Benchmark training performance and investigate incremental pair-count updates.
- Evaluate on larger held-out datasets.
- Eventually compare how different tokenizers affect a small language model trained under controlled conditions.

The goal is to understand each part well enough to explain why it works, where it fails, and what trade-offs are involved before moving to the next improvement.
