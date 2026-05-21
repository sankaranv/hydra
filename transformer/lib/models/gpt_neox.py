"""
HookedGPTNeoX: Pythia/GPT-NeoX wrapper with pyro.deterministic sites.

Architecture overview
---------------------
GPT-NeoX attention uses a combined QKV projection (`query_key_value`) and absolute
dimension indices in its forward pass (.transpose(1, 2) to rearrange to head format,
eager_attention_forward which uses .transpose(1, 2) and key.transpose(2, 3)).
With ChiRho's MultiWorldCounterfactual prepending world dimensions, these absolute
indices break.

GPT-NeoX uses a parallel residual architecture: attn and MLP both operate on the
same pre-norm hidden states and their outputs are summed together with the residual.

Fixes applied
-------------
1. _world_safe_neox_attention_forward: replaces GPTNeoXAttention.forward with a
   version using only relative dims (transpose(-3, -2), transpose(-1, -2)).
   RoPE is applied manually with cos/sin unsqueezed at dim -3 (the head axis)
   rather than dim 1, so it broadcasts correctly under world dims.

2. HookedGPTNeoX.forward: runs embedding and transformer blocks directly, bypassing
   GPTNeoXModel.forward so we control the residual stream sites.

Sites registered (identical naming to HookedGPT2)
-----------------
    resid_pre_0   : embedding output
    head_{L}_{H}  : head H's contribution to the residual stream at layer L
    attn_out_{L}  : full attention output (sum of heads)
    mlp_out_{L}   : MLP block output
    resid_post_{L}: residual after block L

Head contribution uses nn.Linear convention (weight=[d_out, d_in], forward=x@W.T+b):
    head_contribution = pre_proj[..., H*d_head:(H+1)*d_head] @ weight[:, H*d_head:(H+1)*d_head].T
The bias is applied once to the sum (not per head) since it is a shared offset.
"""

import types
from typing import Optional

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXModel
from transformers.models.gpt_neox.modeling_gpt_neox import rotate_half


def _apply_rotary_pos_emb_world_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k tensors that may have leading world dimensions.

    Standard apply_rotary_pos_emb uses unsqueeze_dim=1 (absolute), which breaks
    when world dims are prepended. We insert at dim -3 (the n_heads axis relative
    to the last three dims: [n_heads, seq, d_head]), which is correct regardless of
    how many world/batch dims precede.

    Args:
        q, k : [..., n_heads, seq, d_head] — query and key after head reshape+transpose.
        cos, sin : [batch, seq, rotary_dim] — position embeddings from rotary_emb.

    Returns:
        Rotated q and k with the same shape.
    """
    # Insert head dim for broadcasting: [batch, seq, rotary_dim] → [batch, 1, seq, rotary_dim]
    cos = cos.unsqueeze(-3)
    sin = sin.unsqueeze(-3)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _world_safe_neox_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """GPTNeoXAttention.forward replacement using relative dimension indices.

    Original uses .transpose(1, 2) after the QKV reshape and delegates to
    eager_attention_forward which uses key.transpose(2, 3) and attn_output.transpose(1, 2).
    All of these break under world dims. This replacement uses only relative dims:
    - .transpose(-3, -2) to move to head format
    - .transpose(-1, -2) for the attention score matmul
    - RoPE applied via _apply_rotary_pos_emb_world_safe (unsqueeze at dim -3)

    Ignores past_key_values (no KV cache). Requires attn_implementation='eager'.
    """
    n_heads = self.config.num_attention_heads
    head_size = self.head_size

    # Combined QKV projection: [*world, batch, seq, 3*n_heads*d_head]
    # Reshape to [..., seq, n_heads, 3, d_head] then transpose to [..., n_heads, seq, 3, d_head]
    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(*qkv.shape[:-1], n_heads, 3 * head_size)

    # Split along last dim into 3 tensors of shape [..., seq, n_heads, d_head]
    q, k, v = qkv.split(head_size, dim=-1)

    # [..., seq, n_heads, d_head] → [..., n_heads, seq, d_head]
    q = q.transpose(-3, -2)
    k = k.transpose(-3, -2)
    v = v.transpose(-3, -2)

    # Apply rotary position embeddings (world-safe: unsqueeze at head axis dim -3)
    cos, sin = position_embeddings
    q, k = _apply_rotary_pos_emb_world_safe(q, k, cos, sin)

    # Scaled dot-product attention
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Context: [..., n_heads, seq, d_head]
    attn_output = torch.matmul(attn_weights, v)

    # Merge heads: [..., n_heads, seq, d_head] → [..., seq, n_heads*d_head]
    attn_output = (
        attn_output.transpose(-3, -2)
        .reshape(*attn_output.shape[:-3], attn_output.shape[-2], -1)
        .contiguous()
    )

    attn_output = self.dense(attn_output)
    return attn_output, None


class _NeoXAttentionHook:
    """Forward hook on model.layers[L].attention.dense.

    Decomposes the pre-dense merged output into per-head contributions using
    nn.Linear weight slices (weight=[d_out, d_in], forward=x@W.T+b). Head H's
    contribution to the residual: pre_proj[..., H*d_head:(H+1)*d_head] @ weight[:, H*d_head:(H+1)*d_head].T
    The bias is added to the running sum (not per head) since it is a shared offset.
    """

    def __init__(self, layer_idx: int, n_heads: int, d_head: int) -> None:
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        pre_proj = input[0]  # [*world, batch, seq, n_heads * d_head]
        weight = module.weight  # [d_model, d_model] nn.Linear: x @ W.T
        bias = module.bias  # [d_model] or None

        # Accumulate from bias so rounding matches the native linear forward
        attn_out = (
            bias.clone()
            if bias is not None
            else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        )
        for head_idx in range(self.n_heads):
            start = head_idx * self.d_head
            end = start + self.d_head
            # Weight slice: rows are d_model output, cols are this head's d_head input
            head_contribution = pre_proj[..., start:end] @ weight[:, start:end].T
            head_contribution = pyro.deterministic(
                f"head_{self.layer_idx}_{head_idx}", head_contribution, event_dim=2
            )
            attn_out = attn_out + head_contribution
        return pyro.deterministic(f"attn_out_{self.layer_idx}", attn_out, event_dim=2)


class _NeoXMLPHook:
    """Forward hook on model.layers[L].mlp registering mlp_out_{L}."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        return pyro.deterministic(f"mlp_out_{self.layer_idx}", output, event_dim=2)


class HookedGPTNeoX(nn.Module):
    """Pythia/GPT-NeoX wrapper with pyro.deterministic sites for ChiRho interventions.

    Supports models from the GPT-NeoX family (EleutherAI Pythia). Uses eager
    attention (attn_implementation='eager') and disables the KV cache.

    The parallel residual architecture (attn and MLP applied to the same pre-norm
    hidden states) is handled correctly: both hooks fire on the correct modules and
    the per-layer resid_post site is registered after the combined update.

    Usage::

        model = HookedGPTNeoX("/path/to/pythia-70m")
        trace = pyro.poutine.trace(model).get_trace(input_ids)
    """

    def __init__(self, model_name_or_config: str | GPTNeoXConfig) -> None:
        super().__init__()

        if isinstance(model_name_or_config, GPTNeoXConfig):
            config = model_name_or_config
            config._attn_implementation = "eager"
            self.neox = GPTNeoXModel(config)
        else:
            self.neox = GPTNeoXModel.from_pretrained(
                model_name_or_config,
                attn_implementation="eager",
                local_files_only=True,
            )
            config = self.neox.config

        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.d_model = config.hidden_size
        self.d_head = self.d_model // self.n_heads

        self._patch_attention_forwards()
        self._register_hooks()

    def _patch_attention_forwards(self) -> None:
        """Replace each GPTNeoXAttention.forward with the world-safe version."""
        for layer in self.neox.layers:
            layer.attention.forward = types.MethodType(
                _world_safe_neox_attention_forward, layer.attention
            )

    def _register_hooks(self) -> None:
        """Register forward hooks for per-head contributions and MLP outputs."""
        self._hook_handles = []
        for layer_idx in range(self.n_layers):
            layer = self.neox.layers[layer_idx]
            attn_hook = _NeoXAttentionHook(layer_idx, self.n_heads, self.d_head)
            mlp_hook = _NeoXMLPHook(layer_idx)
            self._hook_handles.append(
                layer.attention.dense.register_forward_hook(attn_hook)
            )
            self._hook_handles.append(layer.mlp.register_forward_hook(mlp_hook))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run GPT-NeoX and register residual stream sites.

        Runs embedding, rotary embedding, and transformer blocks directly to
        avoid GPTNeoXModel.forward's abstraction layers. Manages the causal mask
        explicitly so we control the full data flow.

        Args:
            input_ids: Long tensor of shape [batch, seq].

        Returns:
            hidden_states: Float tensor of shape [*world, batch, seq, d_model].
        """
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embedding + dropout
        hidden_states = self.neox.emb_dropout(self.neox.embed_in(input_ids))
        hidden_states = pyro.deterministic("resid_pre_0", hidden_states, event_dim=2)

        # Rotary position embeddings: (cos, sin) each [batch, seq, rotary_dim]
        position_embeddings = self.neox.rotary_emb(hidden_states, position_ids)

        # Additive causal mask: [1, 1, seq, seq] with -inf above the diagonal
        causal_mask = torch.triu(
            torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=hidden_states.device
            ),
            diagonal=1,
        )

        for layer_idx, layer in enumerate(self.neox.layers):
            # _NeoXAttentionHook and _NeoXMLPHook fire inside layer(), registering
            # head_{L}_{H}, attn_out_{L}, and mlp_out_{L}.
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = pyro.deterministic(
                f"resid_post_{layer_idx}", hidden_states, event_dim=2
            )

        return self.neox.final_layer_norm(hidden_states)
