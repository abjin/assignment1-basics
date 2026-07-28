"""Text generation from a trained language model."""

from __future__ import annotations

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str = "cuda",
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    eos_id = tokenizer.bytes_to_id.get(eos_token.encode("utf-8"))
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = tokens[:, -model.context_length :]
        logits = model(context)[0, -1]

        if temperature == 0:
            next_id = int(logits.argmax())
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_ids = probs.sort(descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                # Keep the smallest prefix whose mass reaches top_p.
                cutoff = int((cumulative < top_p).sum()) + 1
                keep_probs = sorted_probs[:cutoff]
                keep_probs = keep_probs / keep_probs.sum()
                next_id = int(sorted_ids[torch.multinomial(keep_probs, 1)])
            else:
                next_id = int(torch.multinomial(probs, 1))

        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(tokens[0].tolist()[len(ids) :])
