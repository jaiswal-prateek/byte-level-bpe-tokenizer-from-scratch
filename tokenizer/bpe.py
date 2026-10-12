import json
import re
from collections import Counter

class ByteLevelBPETokenizer:
    """
    Educational byte-level BPE tokenizer.
    """

    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.merge_ranks = {}
        self.special_tokens = {}
        self.next_token_id = 0

    def pretokenize(self, text: str) -> list[str]:
        pattern = r" ?[A-Za-z]+(?:'[A-Za-z]+)?| ?[0-9]+|[^\w\s]+|\s+"
        return re.findall(pattern, text)

    def build_piece_corpus(self, texts: list[str]) -> tuple[dict[str, list[int]], Counter]:
        piece_counts = Counter()

        for text in texts:
            pieces = self.pretokenize(text)
            piece_counts.update(pieces)

        piece_corpus = {piece: list(piece.encode("utf-8")) for piece in piece_counts}

        return piece_corpus, piece_counts

    def count_pairs_weighted(self, piece_corpus: dict[str, list[int]], piece_counts: Counter) -> dict[tuple[int, int], int]:
        pair_counts = {}

        for piece, enc in piece_corpus.items():
            for pair in zip(enc, enc[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + piece_counts[piece]

        return pair_counts

    def initialize_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.next_token_id = 256

    def merge_pair(self, tokens: list, pair: tuple[int, int], new_token_id: int) -> list:
        merged = []
        i = 0

        while i < len(tokens):
            if  i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
                
        return merged

    def train(self, texts: list[str], vocab_size: int):
        if not self.vocab:
            self.initialize_vocab()

        max_merges = vocab_size - 256 - len(self.special_tokens)

        piece_corpus, piece_counts = self.build_piece_corpus(texts)

        for rank in range(max_merges):

            pair_counts = self.count_pairs_weighted(piece_corpus, piece_counts)

            if not pair_counts:
                break

            # Select the globally most frequent pair
            best_pair = max(pair_counts, key= pair_counts.get)

            new_token_id = self.next_token_id
            self.next_token_id += 1

            self.merges[best_pair] = new_token_id
            self.merge_ranks[best_pair] = rank

            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            for piece, tokens in piece_corpus.items():
                if best_pair in zip(tokens, tokens[1:]):
                    piece_corpus[piece] = self.merge_pair(tokens, best_pair, new_token_id)   
        # print('\nvocab:', self.vocab)
        # print('\nmerges:', self.merges)
        # print('\nmerge_ranks:', self.merge_ranks)

    def _encode_piece(self, piece: str) -> list[int]:
        tokens = list(piece.encode("utf-8"))

        while True:
            pair_candidates = []

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merge_ranks:
                    pair_candidates.append((self.merge_ranks[pair], pair))

            if not pair_candidates:
                break

            _, best_pair = min(pair_candidates)

            tokens = self.merge_pair(tokens, best_pair, self.merges[best_pair])

        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = []

        pieces = self.pretokenize(text)

        for piece in pieces:
            tokens.extend(self._encode_piece(piece))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        byte_data = [self.vocab[t] for t in tokens]
        byte_data = b''.join(byte_data)
        text = byte_data.decode("utf-8")

        return text
    
    def save(self, path):
        data = {
            "vocab": {
                str(token_id): list(token_bytes)
                for token_id, token_bytes in self.vocab.items()
            },
            "merges": [
                {
                    "pair": list(pair),
                    "token_id": token_id
                }
                for pair, token_id in self.merges.items()
            ],
            "merge_ranks": [
                {
                    "pair": list(pair),
                    "rank": rank
                }
                for pair, rank in self.merge_ranks.items()
            ],
            "special_tokens": self.special_tokens,
            "next_token_id": self.next_token_id
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in data["vocab"].items()
        }

        self.merges = {
            tuple(item["pair"]): item["token_id"]
            for item in data["merges"]
        }

        self.merge_ranks = {
            tuple(item["pair"]): item["rank"]
            for item in data["merge_ranks"]
        }

        self.special_tokens = data["special_tokens"]
        self.next_token_id = data["next_token_id"]