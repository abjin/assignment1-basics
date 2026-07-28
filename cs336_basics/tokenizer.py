"""Byte-level BPE tokenizer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.train_bpe import PAT


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2's mapping from bytes to printable unicode characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        self.bytes_to_id: dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(self.merges)}

        if self.special_tokens:
            # Longest first so overlapping specials match greedily.
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")")
        else:
            self._special_pattern = None

        self._pretoken_cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json

        gpt2_byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = {
            token_id: bytes([gpt2_byte_decoder[c] for c in token_str]) for token_str, token_id in gpt2_vocab.items()
        }
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                if cleaned and len(cleaned.split(" ")) == 2:
                    a, b = cleaned.split(" ")
                    merges.append(
                        (bytes([gpt2_byte_decoder[c] for c in a]), bytes([gpt2_byte_decoder[c] for c in b]))
                    )
        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, pretoken: bytes) -> list[int]:
        tokens = [bytes([b]) for b in pretoken]
        while len(tokens) > 1:
            best_rank = None
            best_idx = None
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                break
            tokens[best_idx : best_idx + 2] = [tokens[best_idx] + tokens[best_idx + 1]]
        return [self.bytes_to_id[token] for token in tokens]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in re.finditer(PAT, text):
            pretoken = match.group()
            ids = self._pretoken_cache.get(pretoken)
            if ids is None:
                ids = self._apply_bpe(pretoken.encode("utf-8"))
                if len(self._pretoken_cache) < 100_000:
                    self._pretoken_cache[pretoken] = ids
            yield from ids

    def _encode_iter(self, text: str) -> Iterator[int]:
        if self._special_pattern is None:
            yield from self._encode_ordinary(text)
            return
        for part in self._special_pattern.split(text):
            if not part:
                continue
            special_id = self.bytes_to_id.get(part.encode("utf-8")) if part in self.special_tokens else None
            if special_id is not None and part in self.special_tokens:
                yield special_id
            else:
                yield from self._encode_ordinary(part)

    def encode(self, text: str) -> list[int]:
        return list(self._encode_iter(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_iter(chunk)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")
