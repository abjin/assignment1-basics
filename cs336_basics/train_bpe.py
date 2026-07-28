"""Byte-level BPE tokenizer training."""

from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Chunk the file into parts that can be counted independently.

    Boundaries are aligned to occurrences of ``split_special_token`` so no
    merge can span a chunk boundary.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _count_pretokens_in_text(text: str, special_tokens: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pattern, text)
    else:
        segments = [text]
    for segment in segments:
        for match in re.finditer(PAT, segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _count_pretokens_in_chunk(args: tuple[str, int, int, list[str]]) -> Counter[bytes]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens_in_text(text, special_tokens)


def _pretokenize(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int) -> Counter[bytes]:
    file_size = os.path.getsize(input_path)
    # Small files are cheaper to process serially than to fork workers for.
    if file_size < 1_000_000 or num_processes <= 1:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
        return _count_pretokens_in_text(text, special_tokens)

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, split_token)

    tasks = [(str(input_path), start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    total: Counter[bytes] = Counter()
    with Pool(num_processes) as pool:
        for counts in pool.imap_unordered(_count_pretokens_in_chunk, tasks):
            total.update(counts)
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocab and ordered merges."""
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 8)

    pretoken_counts = _pretokenize(input_path, special_tokens, num_processes)

    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])

    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []
    if num_merges <= 0:
        return vocab, merges

    # Each distinct pretoken is a word: a list of current tokens plus its count.
    words: list[list[bytes]] = []
    word_counts: list[int] = []
    for pretoken, count in pretoken_counts.items():
        words.append([bytes([b]) for b in pretoken])
        word_counts.append(count)

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}
    for idx, word in enumerate(words):
        count = word_counts[idx]
        for pair in zip(word[:-1], word[1:]):
            pair_counts[pair] += count
            pair_to_words.setdefault(pair, set()).add(idx)

    for _ in range(num_merges):
        if not pair_counts:
            break
        # Most frequent pair; ties broken by lexicographically greater pair.
        best_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == best_count)
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        affected = pair_to_words.pop(best_pair, set())
        del pair_counts[best_pair]

        for idx in affected:
            word = words[idx]
            count = word_counts[idx]

            # Remove this word's contribution to pair statistics.
            for pair in zip(word[:-1], word[1:]):
                if pair == best_pair:
                    continue
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                occurrences = pair_to_words.get(pair)
                if occurrences is not None:
                    occurrences.discard(idx)
                    if not occurrences:
                        del pair_to_words[pair]

            # Merge all occurrences of the pair within the word.
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word

            for pair in zip(new_word[:-1], new_word[1:]):
                pair_counts[pair] += count
                pair_to_words.setdefault(pair, set()).add(idx)

    return vocab, merges
