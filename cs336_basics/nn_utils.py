"""Training utilities: loss, gradient clipping, batching, checkpointing."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy over all batch-like dimensions.

    ``inputs`` has shape (..., vocab_size); ``targets`` has shape (...,).
    """
    inputs = inputs - inputs.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(inputs).sum(dim=-1))
    target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    labels = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    inputs_t = torch.from_numpy(inputs.astype(np.int64)).to(device)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)
    return inputs_t, labels_t


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
