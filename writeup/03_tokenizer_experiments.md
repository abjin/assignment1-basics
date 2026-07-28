# CS336 Assignment 1 — Written Responses: BPE Training & Tokenizer Experiments

Environment: 8-core CPU, 44 GB RAM, no GPU. Python 3 with the `regex` package. Tokenizer
implementation: `/notebooks/cs336_basics/tokenizer.py`; training implementation:
`/notebooks/cs336_basics/train_bpe.py`. Trained artifacts:
`/notebooks/data/tinystories/tokenizer.pkl` (vocab 10,000; special token `<|endoftext|>`).

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

### (a) Compression ratio on 10 sampled TinyStories documents

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

*(OWT part pending: the OpenWebText 32K tokenizer has not been trained yet — this row will be
updated after `train_bpe_expts_owt` is completed.)*

### (b) Tokenizing out-of-domain text with the TinyStories tokenizer

*(Proxy experiment: since the OWT tokenizer/sample is not available yet, we instead encode
other-domain fixture texts with the TinyStories tokenizer; the OWT-sample comparison will be
updated after the OWT tokenizer is trained.)*

| text | domain | bytes | tokens | bytes/token |
|------|--------|------:|-------:|------------:|
| TinyStories sample (10 docs) | in-domain | 8457 | 2084 | **4.058** |
| `tests/fixtures/address.txt` (Gettysburg Address) | formal 19th-c. English | 1468 | 374 | **3.925** |
| `tests/fixtures/german.txt` (German Wikipedia) | German | 594 | 282 | **2.106** |

Compression degrades as we leave the training domain: formal English still compresses
reasonably (3.93 bytes/token, only ~3% worse, since the vocabulary covers common English
words), but German text nearly halves the compression ratio to 2.11 bytes/token — the
tokenizer falls back to short subword fragments and single characters, e.g. the first German
tokens are `['D', 'ie', ' L', 'el', 'and', ' Stan', 'f', 'ord', ' J', 'un', 'i', 'or', ...]`.
A tokenizer trained on one distribution encodes other distributions with many more, shorter
tokens, which wastes sequence length at LM training time.

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
(matching `/notebooks/data/tinystories/train.npy`) instead of ~4.3 GB as int64, and the
smaller memory-mapped arrays also speed up data loading during LM training. A signed `int16`
would not work (max 32,767 < 65,535 and wastes the sign bit), and `uint8` is too small
(max 255), so `uint16` is the smallest dtype that safely holds the vocabulary.
