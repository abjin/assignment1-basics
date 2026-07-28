"""Train a BPE tokenizer on a corpus and encode it into uint16 token arrays.

Usage:
    python -m cs336_basics.prepare_data \
        --train data/TinyStoriesV2-GPT4-train.txt \
        --valid data/TinyStoriesV2-GPT4-valid.txt \
        --vocab-size 10000 --out-dir data/tinystories
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


def encode_file_to_array(tokenizer: Tokenizer, input_path: str, output_path: str) -> int:
    ids = []
    with open(input_path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)
    return len(arr)


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

    tokenizer = Tokenizer(vocab, merges, special_tokens)

    for split, path in [("train", args.train), ("valid", args.valid)]:
        out_path = out_dir / f"{split}.npy"
        if out_path.exists():
            print(f"{split}: already encoded, skipping")
            continue
        start = time.time()
        n = encode_file_to_array(tokenizer, path, str(out_path))
        print(f"{split}: {n} tokens in {time.time() - start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
