# CS336 Assignment 1 — Written Responses: BPE Training & Tokenizer Experiments

Environment: 8-core CPU, 44 GB RAM, no GPU. Python 3 with the `regex` package. Tokenizer
implementation: `/notebooks/cs336_basics/tokenizer.py`; training implementation:
`/notebooks/cs336_basics/train_bpe.py`. Trained artifacts:
`/notebooks/data/tinystories/tokenizer.pkl` (vocab 10,000; special token `<|endoftext|>`) and
`/notebooks/data/owt/tokenizer.pkl` (vocab 32,000; special token `<|endoftext|>`).

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### (a) Time, memory, and the longest token

Training a byte-level BPE tokenizer on the full TinyStories training set
(`TinyStoriesV2-GPT4-train.txt`, 2.23 GB) with `vocab_size=10000` and the `<|endoftext|>`
special token took **113.2 s** (from `/notebooks/data/prepare.log`; an instrumented re-run
with `resource.getrusage` reproduced this at 130.7 s wall time). Peak memory was small: the
main process peaked at **0.09 GB RSS** and the largest of the 8 pre-tokenization workers at
**0.43 GB RSS**, i.e. roughly **3–4 GB aggregate** across all processes at peak — far under
the 30 min / 30 GB budget. Memory stays low because workers stream `<|endoftext|>`-aligned
file chunks and return only `Counter` objects of distinct pre-tokens, so the 2.23 GB corpus
is never held in memory at once. Pre-tokenization ran in 8 parallel worker processes over `<|endoftext|>`-aligned
file chunks; the merge loop ran serially over the aggregated pre-token counts.

The longest tokens in the vocabulary (found by sorting `vocab.values()` by byte length):

```
len=15  b' accomplishment'
len=15  b' disappointment'
len=15  b' responsibility'
len=14  b' uncomfortable'
len=14  b' compassionate'
```

The longest token is 15 bytes, e.g. `b' accomplishment'` (a leading space plus a whole word).
This makes sense: TinyStories is simple, clean, GPT-4-generated children's-story English, so
the most frequent long sequences are common whole English words (with their preceding space,
because the GPT-2 pre-tokenizer attaches the leading space to each word), and words like
"accomplishment" recur constantly in the dataset's moral-lesson endings.

### (b) Profiling: what takes the most time?

**Pre-tokenization is the bottleneck in total compute; after parallelizing it, wall-clock time
is split between parallel pre-tokenization and the serial merge loop.** Pre-tokenization cost
scales linearly with corpus size (regex matching over all 2.23 GB), whereas the merge loop
scales only with the number of *distinct* pre-tokens (tens of thousands for TinyStories),
independent of corpus size. A phase-timing measurement on the 22.5 MB validation set (serial,
1 process) shows the split at small scale:

```
pretokenize (serial): 7.6 s   (13,111 distinct pretokens)
merge loop (9,743 merges): 9.0 s
```

Scaling to the ~100x larger training set, serial pre-tokenization would take on the order of
~760 s while the merge loop grows only modestly (it depends on distinct pre-tokens, not raw
bytes). With 8 worker processes, pre-tokenization drops to roughly ~85–90 s, which together
with the ~20–25 s serial merge loop matches the observed 113 s total. So the single most
expensive stage is regex pre-tokenization, and it is also the stage that parallelizes; the
serial merge loop (dominated by re-scanning `pair_counts` for the max each iteration and
updating affected words) is the residual cost that remains after parallelization.

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

### (a) The longest token in the OWT vocabulary

A byte-level BPE tokenizer was trained on the OpenWebText training set
(`/notebooks/data/owt_train.txt`, 11.9 GB) with `vocab_size=32000` and the `<|endoftext|>`
special token, and serialized to `/notebooks/data/owt/tokenizer.pkl` (32,000 vocab entries,
31,743 merges). The longest tokens (by byte length):

```
len=64  b'\xc3\x83\xc3\x82' * 16          # "ÃÂ" repeated 16 times (mojibake)
len=64  b'-' * 64                          # 64 hyphens
len=48  b'\xe2\x80\x94' * 16               # em-dash "—" repeated 16 times
len=32  b'-'*32, b'_'*32, b'='*32, b'.'*32, b'*'*32, and "ÃÂ"*8
```

The longest token is 64 bytes: a tie between `'ÃÂ'` repeated 16 times and a run of 64
hyphens. This makes sense for scraped web text: `ÃÂ` is a classic UTF-8 double-encoding
(mojibake) artifact that appears in long runs on badly-encoded pages, and hyphen/underscore/
equals runs are ASCII separator lines — both recur verbatim thousands of times, so BPE keeps
merging them into ever-longer tokens. (The longest *linguistic* tokens are 19 bytes:
`b' disproportionately'`, `b' telecommunications'`.)

### (b) TinyStories vs. OpenWebText tokenizer

The TinyStories 10K vocabulary is dominated by whole common English words — its longest
tokens are clean words like `b' accomplishment'` (15 bytes) — reflecting a small, simple,
homogeneous GPT-4-generated vocabulary, while the OWT 32K vocabulary additionally captures
web-specific structure: separator-line runs, mojibake artifacts, URL/code fragments, and a
much larger inventory of rarer and longer words (e.g. `b' disproportionately'`) thanks to
both the noisier domain and the 3.2x larger vocabulary budget. Consequently the OWT
tokenizer compresses web text much better (4.38 vs 2.95 bytes/token on the OWT sample, see
below) and even compresses TinyStories text almost as well as the in-domain tokenizer
(3.94 vs 4.06 bytes/token), whereas the TinyStories tokenizer generalizes poorly outside
its narrow domain.

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

Measurement script (excerpted; run with system `python3`, seed 42):

```python
import pickle, random, time
from cs336_basics.tokenizer import Tokenizer

blob = pickle.load(open("/notebooks/data/tinystories/tokenizer.pkl", "rb"))
tok = Tokenizer(blob["vocab"], blob["merges"], blob["special_tokens"])

text = open("/notebooks/data/TinyStoriesV2-GPT4-valid.txt").read()
docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]
random.seed(42)
sample = random.sample(docs, 10)

# OWT sampling: read only the first 8 MB of the 290 MB validation file,
# split on <|endoftext|>, drop the trailing partial document (1,586 docs),
# then random.sample(owt_docs, 10) with seed 42.
total_bytes = sum(len(d.encode("utf-8")) for d in sample)
total_tokens = sum(len(tok.encode(d)) for d in sample)
print(total_bytes / total_tokens)  # compression ratio, bytes/token

# throughput: warm up on 2 MB, then time a fresh 5 MB slice
_ = tok.encode(text[:2_000_000])
bench = text[2_000_000:7_000_000]
nb = len(bench.encode("utf-8"))
t0 = time.perf_counter(); ids = tok.encode(bench); dt = time.perf_counter() - t0
print(nb / dt)  # bytes/sec
```

### (a) Compression ratio on 10 sampled TinyStories and OpenWebText documents

Sampling 10 documents (seed 42) from the TinyStories validation set (27,630 documents) and
encoding them with the 10K TinyStories tokenizer:

| doc | bytes | tokens | bytes/token |
|----:|------:|-------:|------------:|
| 1 | 687 | 171 | 4.018 |
| 2 | 2666 | 668 | 3.991 |
| 3 | 603 | 148 | 4.074 |
| 4 | 448 | 101 | 4.436 |
| 5 | 797 | 197 | 4.046 |
| 6 | 565 | 128 | 4.414 |
| 7 | 710 | 173 | 4.104 |
| 8 | 814 | 210 | 3.876 |
| 9 | 630 | 158 | 3.987 |
| 10 | 537 | 130 | 4.131 |
| **total** | **8457** | **2084** | **4.058** |

The TinyStories 10K tokenizer achieves a compression ratio of **~4.06 bytes/token** on
in-domain text (per-document range 3.88–4.44). All encodings round-trip exactly through
`decode`.

Sampling 10 documents (seed 42) from the first 8 MB of the OWT validation set (1,586
documents) and encoding them with the 32K OWT tokenizer:

| doc | bytes | tokens | bytes/token |
|----:|------:|-------:|------------:|
| 1 | 2565 | 568 | 4.516 |
| 2 | 2276 | 544 | 4.184 |
| 3 | 8121 | 1733 | 4.686 |
| 4 | 3371 | 827 | 4.076 |
| 5 | 13097 | 2834 | 4.621 |
| 6 | 3104 | 682 | 4.551 |
| 7 | 26310 | 5596 | 4.702 |
| 8 | 2172 | 493 | 4.406 |
| 9 | 22852 | 5962 | 3.833 |
| 10 | 3440 | 692 | 4.971 |
| **total** | **87308** | **19931** | **4.381** |

The OWT 32K tokenizer achieves **~4.38 bytes/token** on in-domain web text (per-document
range 3.83–4.97) — slightly better than TinyStories' 4.06 despite the harder domain, because
of its 3.2x larger vocabulary. All encodings round-trip exactly through `decode`. (For
reference, the OWT tokenizer also encodes the TinyStories sample at 3.94 bytes/token —
cross-domain, but nearly as good as the in-domain TinyStories tokenizer.)

### (b) Tokenizing the OWT sample with the TinyStories tokenizer

Encoding the same 10 OWT documents with the TinyStories 10K tokenizer instead of the OWT
32K tokenizer:

| text | tokenizer | bytes | tokens | bytes/token |
|------|-----------|------:|-------:|------------:|
| OWT sample (10 docs) | OWT 32K (in-domain) | 87308 | 19931 | **4.381** |
| OWT sample (10 docs) | TinyStories 10K | 87308 | 29604 | **2.949** |
| TinyStories sample (10 docs) | TinyStories 10K (in-domain) | 8457 | 2084 | **4.058** |

The compression ratio drops from 4.38 to **2.95 bytes/token** (per-document range 2.58–3.36),
i.e. the same text costs **~49% more tokens** (29,604 vs 19,931). Qualitatively, the
TinyStories tokenizer shatters web-domain vocabulary into short subword fragments — e.g. the
first OWT document opens with `['B', 're', 'aking', ' Ne', 'w', 's', ' Em', 'ails', ...]`
where the OWT tokenizer produces `['Breaking', ' News', ' Emails', ...]` — because words like
"Breaking", "Emails", or "alerts" simply never appear in children's stories. A tokenizer
trained on one distribution encodes other distributions with many more, shorter tokens,
which wastes sequence length at LM training time.

Supplementary fixture measurements with the TinyStories tokenizer show the same trend
worsening with distance from the training domain: the Gettysburg Address (formal English)
still gets 3.93 bytes/token, but German Wikipedia text collapses to 2.11 bytes/token.

### (c) Throughput and time to tokenize the Pile

Measured by encoding a fresh 5 MB slice of the TinyStories validation file (after a 2 MB
warm-up so the pre-token cache is in its steady state, matching long-run behavior):

- **Warm throughput: 5.12 MB/s** (≈1.24 M tokens/s, single process; cold-cache: 4.25 MB/s).
- The Pile (825 GB) at 5.12 MB/s: 825e9 / 5.12e6 ≈ **161,000 s ≈ 44.8 hours (~1.9 days)** on
  a single process; tokenization is embarrassingly parallel across documents, so with 8
  processes this drops to roughly **~5.6 hours**.

### (d) Why is uint16 an appropriate dtype for serialized token IDs?

`uint16` represents integers 0–65,535, which covers every ID in a 10K (and a 32K, and even a
64K) vocabulary, while using only 2 bytes per token — half of `int32` and a quarter of
`int64`. For the 540.8 M-token TinyStories training encoding this is ~1.08 GB on disk
(matching `/notebooks/data/tinystories/train.npy`), and for the 2.73 B-token OWT training
encoding ~5.45 GB (matching `/notebooks/data/owt/train.npy`) instead of ~21.8 GB as int64; the
smaller memory-mapped arrays also speed up data loading during LM training. A signed `int16`
would not work (max 32,767 < 65,535 and wastes the sign bit), and `uint8` is too small
(max 255), so `uint16` is the smallest dtype that safely holds the vocabulary.
