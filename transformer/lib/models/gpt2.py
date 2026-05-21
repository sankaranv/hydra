"""
HookedGPT2: GPT-2 wrapper with pyro.deterministic sites for ChiRho interventions.

Architecture overview
---------------------
GPT-2 uses Conv1D internally and its original attention forward relies on
.split(dim=2) and .transpose(1, 2), which hardcode absolute tensor dimensions.
When ChiRho's MultiWorldCounterfactual injects leading world dimensions these
calls break. Additionally, GPT2Model.forward calls hidden_states.view(-1, ...) at
the end, which collapses world and batch dimensions.

Fixes applied
-------------
1. _world_safe_attention_forward: replaces GPT2Attention.forward with an equivalent
   that uses only relative dimensions (dim=-1, transpose(-3, -2)) and avoids
   calling eager_attention_forward (whose internal transpose(1, 2) is also absolute).
   All models are loaded with attn_implementation='eager' to ensure a single code path.

2. HookedGPT2.forward: runs the transformer step-by-step instead of delegating to
   GPT2Model.forward, skipping its final hidden_states.view(-1, seq, d) which would
   merge world and batch dimensions.

Sites registered
----------------
For each layer L in range(n_layers):
    head_{L}_{H}  : per-head contribution to residual stream, shape [*world, batch, seq, d_model]
    attn_out_{L}  : full attention output, shape [*world, batch, seq, d_model]
    mlp_out_{L}   : MLP output, shape [*world, batch, seq, d_model]
    resid_post_{L}: residual after block L, shape [*world, batch, seq, d_model]

resid_pre_0 = embedding output (token + position).

Head H's contribution uses Conv1D convention: weight=[d_model, d_model], forward=x@W+b.
    head_contribution = pre_proj[..., H*d_head:(H+1)*d_head] @ weight[H*d_head:(H+1)*d_head, :]
"""

import types
from typing import Optional

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


def _world_safe_attention_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """GPT2Attention.forward replacement using relative dimension indices.

    Differs from the original only in:
    - .split(split_size, dim=-1) instead of dim=2
    - .transpose(-3, -2) instead of .transpose(1, 2)
    - manual attention computation instead of eager_attention_forward (which
      does its own .transpose(1, 2) that breaks world-indexed tensors)

    Requires attn_implementation='eager'. Ignores past_key_values (no KV cache).
    """
    q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=-1)

    # [*world, batch, seq, d_model] → [*world, batch, seq, n_heads, d_head]
    # → [*world, batch, n_heads, seq, d_head]
    shape = (*q.shape[:-1], -1, self.head_dim)
    q = q.view(shape).transpose(-3, -2)
    k = k.view(shape).transpose(-3, -2)
    v = v.view(shape).transpose(-3, -2)

    # Scaled dot-product attention: [*world, batch, n_heads, seq, seq]
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Context: [*world, batch, n_heads, seq, d_head]
    attn_output = torch.matmul(attn_weights, v)

    # Merge heads: [*world, batch, n_heads, seq, d_head] → [*world, batch, seq, d_model]
    attn_output = (
        attn_output.transpose(-3, -2)
        .reshape(*attn_output.shape[:-3], attn_output.shape[-2], -1)
        .contiguous()
    )

    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, None


class _AttentionHook:
    """Forward hook on model.h[L].attn.c_proj.

    Decomposes the pre-c_proj merged attention output into per-head contributions
    using Conv1D weight slices (weight=[d_model, d_model], forward=x@W+b), registers
    each as pyro.deterministic, and reconstructs attn_out from the (possibly patched)
    head contributions.

    Head H's contribution: pre_proj[..., H*d_head:(H+1)*d_head] @ weight[H*d_head:(H+1)*d_head, :]
    The c_proj bias is added to the sum (not per-head) because it is a shared offset.
    Accumulation starts from bias to match the native c_proj forward's rounding order.
    """

    def __init__(self, layer_idx: int, n_heads: int, d_head: int) -> None:
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        pre_proj = input[0]  # [*world, batch, seq, d_model]
        weight = module.weight  # [d_model, d_model] Conv1D: x @ weight
        bias = module.bias  # [d_model]

        attn_out = bias.clone()
        for head_idx in range(self.n_heads):
            start = head_idx * self.d_head
            end = start + self.d_head
            head_contribution = pre_proj[..., start:end] @ weight[start:end, :]
            head_contribution = pyro.deterministic(
                f"head_{self.layer_idx}_{head_idx}", head_contribution, event_dim=2
            )
            attn_out = attn_out + head_contribution
        return pyro.deterministic(f"attn_out_{self.layer_idx}", attn_out, event_dim=2)


class _MLPHook:
    """Forward hook on model.h[L].mlp registering mlp_out_{L} as a pyro.deterministic site."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        return pyro.deterministic(f"mlp_out_{self.layer_idx}", output, event_dim=2)


class HookedGPT2(nn.Module):
    """GPT-2 wrapper with pyro.deterministic sites for ChiRho interventions.

    Loads a GPT-2 model (or uses a provided GPT2Config for a tiny model) and
    registers intermediate activations as named sites. All sites use event_dim=2
    since the natural event tensor is [seq, d_model].

    Loaded with attn_implementation='eager' to ensure a deterministic, inspectable
    code path. The KV cache is disabled in forward() (use_cache=False).

    Usage::

        model = HookedGPT2("gpt2")
        trace = pyro.poutine.trace(model).get_trace(input_ids)
    """

    def __init__(self, model_name_or_config: str | GPT2Config = "gpt2") -> None:
        super().__init__()

        if isinstance(model_name_or_config, GPT2Config):
            config = model_name_or_config
            config._attn_implementation = "eager"
            self.gpt2 = GPT2Model(config)
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                model_name_or_config,
                attn_implementation="eager",
                local_files_only=True,
            )
            config = self.gpt2.config

        self.n_layers = config.n_layer
        self.n_heads = config.n_head
        self.d_model = config.n_embd
        self.d_head = self.d_model // self.n_heads

        self._patch_attention_forwards()
        self._register_hooks()

    def _patch_attention_forwards(self) -> None:
        """Replace each GPT2Attention.forward with the world-safe version."""
        for block in self.gpt2.h:
            block.attn.forward = types.MethodType(
                _world_safe_attention_forward, block.attn
            )

    def _register_hooks(self) -> None:
        """Register forward hooks for per-head contributions and MLP outputs."""
        self._hook_handles = []
        for layer_idx in range(self.n_layers):
            attn_hook = _AttentionHook(layer_idx, self.n_heads, self.d_head)
            mlp_hook = _MLPHook(layer_idx)
            self._hook_handles.append(
                self.gpt2.h[layer_idx].attn.c_proj.register_forward_hook(attn_hook)
            )
            self._hook_handles.append(
                self.gpt2.h[layer_idx].mlp.register_forward_hook(mlp_hook)
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run GPT-2 and register residual stream sites.

        Bypasses GPT2Model.forward's final hidden_states.view(-1, seq, d_model),
        which would collapse world and batch dimensions under MultiWorldCounterfactual.
        Runs embedding and transformer blocks directly with an explicit causal mask.

        Args:
            input_ids: Long tensor of shape [batch, seq].

        Returns:
            hidden_states: Float tensor of shape [*world, batch, seq, d_model].
        """
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = self.gpt2.drop(inputs_embeds + position_embeds)

        hidden_states = pyro.deterministic("resid_pre_0", hidden_states, event_dim=2)

        # Additive causal mask: [1, 1, seq, seq] with -inf above the diagonal
        causal_mask = torch.triu(
            torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=hidden_states.device
            ),
            diagonal=1,
        )

        for layer_idx, block in enumerate(self.gpt2.h):
            # _AttentionHook and _MLPHook fire inside block(), registering
            # head_{L}_{H}, attn_out_{L}, and mlp_out_{L}.
            hidden_states = block(
                hidden_states,
                past_key_values=None,
                attention_mask=causal_mask,
                use_cache=False,
            )
            hidden_states = pyro.deterministic(
                f"resid_post_{layer_idx}", hidden_states, event_dim=2
            )

        return self.gpt2.ln_f(hidden_states)
