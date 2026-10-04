# Tokenization Fundamentals

## 1. Character Tokenization

Started with a very simple character-level tokenizer.

Example:

```text
hello, → h e l l o ,
```

Each unique character gets an ID using `stoi`, and `itos` is used to convert the IDs back.

Example:

```text
hello, → [7, 4, 9, 9, 11, 1]
```

### What I observed

Character tokenization is very simple and can represent any character that exists in the vocabulary.

But if a new character is encountered:

```text
hello@
```

and `@` was not present when the vocabulary was created, encoding fails with:

```text
KeyError
```

So the first problem is the **unknown character problem**.

Also, character-level tokenization produces many tokens for normal text.

---

## 2. Word Tokenization

Next I tried word-level tokenization.

For example:

```text
hello world, we are living
```

becomes something like:

```text
hello
world
,
we
are
living
```

Each word gets its own token ID.

### What I observed

Word tokenization gives much shorter sequences compared with character tokenization.

But it creates another problem.

There can be a huge number of possible words.

For example:

```text
play
playing
played
player
players
playground
```

can all become separate vocabulary entries.

Also, a word that was not present during training causes a lookup failure.

I then added an `<unk>` token.

For example:

```text
I am king
```

if these words are not in the vocabulary can become:

```text
<unk> <unk> <unk>
```

This avoids the error, but information is lost.

---

## 3. Why Subword Tokenization?

I then started thinking about words such as:

```text
play
playing
played
player
players
playground
```

There is clearly some reusable structure:

```text
play
ing
ed
er
s
ground
```

So instead of treating every complete word as a separate token, we can represent words using smaller reusable pieces.

For example:

```text
players → play + er + s

playing → play + ing

played → play + ed

playground → play + ground
```

I manually created a small subword vocabulary to understand this idea.

### What I observed

This can reduce the number of unique words that need to be stored.

But manually deciding the subword pieces does not scale.

The next question became:

> How can we automatically discover these reusable pieces?

This led to BPE.

---

# 4. BPE — First Implementation

I started with a very small corpus:

```python
corpus = [
    "low",
    "lower",
    "lowest",
    "low",
    "lower",
]
```

Initially, every word was represented using individual characters.

For example:

```text
lower → l o w e r
```

The basic BPE algorithm I implemented was:

```text
1. Count all adjacent token pairs.
2. Find the most frequent pair.
3. Merge that pair.
4. Update the tokenized corpus.
5. Count pairs again.
6. Repeat.
```

For example, the learned merges were:

```text
l + o → lo
lo + w → low
low + e → lowe
lowe + r → lower
```

So the vocabulary gradually grows:

```text
characters
    ↓
lo
    ↓
low
    ↓
lowe
    ↓
lower
```

---

# 5. Understanding BPE Merge Order

I then inspected the learned merge rules manually.

For example:

```text
('l', 'o')
('lo', 'w')
('e', 'r')
('low', 'er')
```

The important observation was that BPE is not simply finding the top N pairs once.

It works iteratively.

At every iteration:

```text
current tokenized corpus
        ↓
count current adjacent pairs
        ↓
find most frequent pair
        ↓
merge it
        ↓
current tokenized corpus changes
        ↓
count pairs again
```

So the pair counts can change after every merge.

For example:

```text
l o w e r
```

After:

```text
l + o → lo
```

we now have:

```text
lo w e r
```

Then:

```text
lo + w → low
```

gives:

```text
low e r
```

Then another merge can create:

```text
low + er → lower
```

So later merges can operate on tokens that were created by earlier merges.

This helped me understand why **merge order matters**.

---

# 6. BPE Encoding

After training the BPE merges, I implemented encoding.

The important idea is:

```text
training:
corpus → learned merge rules

encoding:
new word → start from characters → apply learned merge rules
```

For example:

```text
lowerstlo
```

could become:

```text
lower + s + t + lo
```

Then these pieces are converted into token IDs using `stoi`.

Decoding does the reverse:

```text
token IDs
    ↓
token pieces using itos
    ↓
join pieces
    ↓
original text
```

So the complete pipeline is:

```text
TRAINING

corpus
  ↓
character representation
  ↓
pair counts
  ↓
best pair
  ↓
merge
  ↓
repeat
  ↓
merge rules + vocabulary


ENCODING

text
  ↓
characters
  ↓
apply learned merges
  ↓
BPE pieces
  ↓
token IDs


DECODING

token IDs
  ↓
token pieces
  ↓
join pieces
  ↓
text
```

---

# 7. BPE With a Richer Corpus

I then used a richer corpus containing repeated patterns:

```python
corpus = [
    "low",
    "lower",
    "lowest",
    "playing",
    "played",
    "player",
    "walking",
    "talking",
    "flower",
    "slower",
    "umber",
]
```

This produced more interesting merge rules.

For example:

```text
l + o → lo
lo + w → low
e + r → er
low + er → lower
```

and later:

```text
p + l → pl
pl + a → pla
pla + y → play
```

and:

```text
i + n → in
in + g → ing
```

This showed how BPE can discover reusable pieces automatically.

For example:

```text
playing → play + ing
```

instead of representing it as:

```text
p + l + a + y + i + n + g
```

---

# 8. Testing New Words

I tested words that were not necessarily directly present in the training corpus:

```text
lowering
flowers
flowing
walkers
talkers
playfully
slowly
```

The important observation was that BPE can combine smaller learned pieces to represent new words.

For example:

```text
lowering → lower + ing
```

when those pieces have been learned.

This is one of the main advantages of subword tokenization over pure word tokenization.

---

# 9. Vocabulary Size vs Token Efficiency

I then experimented with different numbers of BPE merges.

For example:

```text
num_merges = 5
num_merges = 10
num_merges = 15
...
num_merges = 50
```

I measured:

```text
vocabulary size
average number of tokens per word
```

The results showed:

```text
more merges
    ↓
larger vocabulary
    ↓
larger reusable pieces
    ↓
fewer tokens per word
```

For the experiment:

```text
5 merges  → avg 4.9 tokens/word
10 merges → avg 3.7
15 merges → avg 3.1
20 merges → avg 2.2
25 merges → avg 1.8
30 merges → avg 1.6
35 merges → avg 1.3
40 merges → avg 1.2
45 merges → avg 1.1
50 merges → avg 1.0
```

The important observation was that increasing the number of merges eventually starts producing complete words as tokens.

For example:

```text
lowering
```

can eventually become one token instead of:

```text
lower + ing
```

So there is a trade-off:

```text
fewer merges
    ↓
smaller vocabulary
    ↓
more tokens per word
    ↓
more reusable subword pieces


more merges
    ↓
larger vocabulary
    ↓
fewer tokens per word
    ↓
more corpus-specific / word-level tokens
```

I also observed diminishing returns as the number of merges increased.

---

# 10. Important Limitation of My Current BPE

The current implementation is still a toy BPE implementation.

It starts from the characters that exist in the training corpus.

Therefore, if a completely new character is encountered during encoding, it cannot be represented.

This was visible earlier with a word such as:

```text
lowering
```

when the training corpus did not contain:

```text
i
n
g
```

The tokenizer could not find those characters in `stoi`.

So I understood an important limitation:

> BPE can reuse learned subwords, but the tokenizer still needs a base vocabulary that can represent arbitrary input.

My current implementation also does not yet handle:

```text
spaces
punctuation
arbitrary Unicode text
special tokens
```

These are things I will handle later.

---

# 11. What I Understand About BPE Now

The most important thing I understand is that BPE is fundamentally an **iterative merging algorithm**.

It does not start with a predefined list of meaningful subwords such as:

```text
play
ing
ed
er
```

Instead, it discovers reusable pieces from the data.

The basic algorithm is:

```text
Start with characters

        ↓

Count adjacent token pairs

        ↓

Select the most frequent pair

        ↓

Merge the pair

        ↓

Recalculate pair frequencies

        ↓

Repeat
```

The output of training is mainly:

```text
learned merge rules
+
vocabulary
```

The merge rules are important because their order determines how new text is segmented.

During encoding:

```text
new text
    ↓
start from base tokens
    ↓
apply learned merges in order
    ↓
BPE pieces
    ↓
token IDs
```

So I now understand the basic BPE algorithm rather than only using a tokenizer library.

---

# 12. Next Step

The current notebook is mainly for understanding the BPE algorithm manually.

The next step is to move this implementation into reusable code:

```text
tokenizer/bpe.py
```

Then I can start improving the tokenizer beyond the toy implementation.

The next major problem to solve is:

```text
How do we make the tokenizer capable of representing arbitrary text?
```

This will lead to:

```text
BPE
  ↓
base vocabulary / text boundaries
  ↓
bytes
  ↓
byte-level BPE
```

Before moving to byte-level BPE, I want to understand how the current implementation can be turned into a clean reusable tokenizer.