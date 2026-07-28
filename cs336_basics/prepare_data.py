"""Train a BPE tokenizer on a corpus and encode it into uint16 token arrays.

Usage:
    python -m cs336_basics.prepare_data \
        --train data/TinyStoriesV2-GPT4-train.txt \
        --valid data/TinyStoriesV2-GPT4-valid.txt \
        --vocab-size 10000 --out-dir data/tinystories
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries, train_bpe

_worker_tokenizer: Tokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    with open(tokenizer_path, "rb") as f:
        saved = pickle.load(f)
    _worker_tokenizer = Tokenizer(saved["vocab"], saved["merges"], saved["special_tokens"])


def _encode_span(args: tuple[str, int, int]) -> np.ndarray:
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(_worker_tokenizer.encode(text), dtype=np.uint16)


def encode_file_to_array(
    tokenizer_path: str, input_path: str, output_path: str, num_processes: int | None = None
) -> int:
    """Encode a corpus into a uint16 token array, in parallel over document-aligned chunks."""
    if num_processes is None:
        num_processes = max(1, min((os.cpu_count() or 1) - 2, 6))

    num_chunks = max(num_processes * 4, min(1024, os.path.getsize(input_path) // 4_000_000 + 1))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    parts = []
    with Pool(num_processes, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        for arr in pool.imap(_encode_span, tasks):
            parts.append(arr)
    total = np.concatenate(parts)
    np.save(output_path, total)
    return len(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = ["<|endoftext|>"]

    tokenizer_path = out_dir / "tokenizer.pkl"
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as f:
            saved = pickle.load(f)
        vocab, merges = saved["vocab"], saved["merges"]
        print(f"Loaded existing tokenizer with {len(vocab)} tokens")
    else:
        start = time.time()
        vocab, merges = train_bpe(args.train, args.vocab_size, special_tokens)
        print(f"BPE training took {time.time() - start:.1f}s, vocab size {len(vocab)}")
        with open(tokenizer_path, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)

    for split, path in [("train", args.train), ("valid", args.valid)]:
        out_path = out_dir / f"{split}.npy"
        if out_path.exists():
            print(f"{split}: already encoded, skipping")
            continue
        start = time.time()
        n = encode_file_to_array(str(tokenizer_path), path, str(out_path))
        print(f"{split}: {n} tokens in {time.time() - start:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
