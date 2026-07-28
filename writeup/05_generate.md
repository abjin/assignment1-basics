# Problem (generate): Generate text

## Setup

Generated with the base TinyStories model (17M non-embedding params, 20,000 steps,
batch 64, context 256, final validation loss **1.374**) using the decoder in
`cs336_basics/decoding.py` with `temperature=0.8`, `top_p=0.9`, prompt `"Once upon a time"`.

Reproduce:

```sh
python3 -m cs336_basics.generate --run-dir runs/tinystories-base --checkpoint final.pt \
    --tokenizer data/tinystories/tokenizer.pkl --prompt "Once upon a time" \
    --max-new-tokens 300 --temperature 0.8 --top-p 0.9
```

## Output (stopped at first `<|endoftext|>`, 180 tokens)

> Once upon a time, there was a little boy named Tim. Tim loved to play with his toys. One day, he found a new toy in his room. It was a big, flexible snake. Tim liked to twist the snake in a fun way.
> Tim showed the snake to his friend, Sam. Sam said, "Wow! That snake is so cool! Can we play with it?" Tim and Sam played with the snake all day. They twisted it and made it twist.
> At the end of the day, Tim and Sam were tired. They sat down and looked at the flexible snake. Tim said, "I like to twist this snake!" Sam agreed, "Me too! We are happy to play with it!" And they both went to sleep, excited to play with their new toy.
> `<|endoftext|>`

The raw dump is in `05_generation_sample.txt`.

## Comment on fluency

The output is fluent and fully coherent for the TinyStories domain: grammatical
sentences, a consistent cast (Tim, Sam, the snake), correct dialogue punctuation, a
beginning–middle–end story arc, and a clean self-terminated ending via `<|endoftext|>`.
Minor weaknesses are repetitive phrasing ("twist the snake" recurs) and the slightly
odd adjective choice "flexible snake" for a toy.

Two factors that strongly affect output quality:

1. **Decoding hyperparameters** — temperature and top-p control the fluency/diversity
   trade-off: at temperature 1.0 with no nucleus truncation the low-probability tail
   introduces incoherent words, while temperature 0.8 + top-p 0.9 prunes that tail and
   yields much more fluent text (temperature 0 instead becomes repetitive and dull).
2. **Model capacity vs. domain and training length** — a 17M-parameter model trained for
   327M tokens is far past the point of fluent grammar on a simple, narrow-vocabulary
   domain like TinyStories, but the same budget on broader text (e.g., OpenWebText)
   yields much less coherent samples; validation loss (1.374 here) tracks this closely.
