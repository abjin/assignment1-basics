"""Transformer language model components."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight.to(torch.float32)
        return result.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(round(d_model * 8 / 3 / 64)) * 64 or 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Non-gated FFN used for the SwiGLU-vs-SiLU ablation."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE dimension must be even"
        self.d_k = d_k
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k / 2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions]  # (..., seq_len, d_k / 2)
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated


def softmax(x: Tensor, dim: int) -> Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ V


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *batch_dims, seq_len, _ = x.shape

        def split_heads(t: Tensor) -> Tensor:
            return t.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)

        q = split_heads(self.q_proj(x))  # (..., num_heads, seq_len, head_dim)
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        out = out.transpose(-3, -2).reshape(*batch_dims, seq_len, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.post_norm = post_norm
        norm_cls = RMSNorm if use_rmsnorm else nn.Identity
        self.ln1 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.ln2 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: Tensor) -> Tensor:
        if self.post_norm:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = (
            RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
            if use_rope
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=rope,
                    use_rmsnorm=use_rmsnorm,
                    post_norm=post_norm,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
