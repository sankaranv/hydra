"""
HookedLlama: shared wrapper for Llama and Qwen2 model families.

Architecture overview
---------------------
Llama 3.1 and Qwen2 share nearly identical attention architectures:
- Separate q_proj, k_proj, v_proj (nn.Linear) with GQA (n_kv_heads <= n_heads)
- Output projection: o_proj (nn.Linear, weight=[d_model, d_model])
- SwiGLU MLP with gate_proj, up_proj, down_proj
- RoPE position embeddings

Both families use .transpose(1, 2) in their attention forward and delegate to
eager_attention_forward which uses absolute dim indices (.transpose(2, 3) and
.transpose(1, 2)). Under ChiRho's MultiWorldCounterfactual these break.

Fixes applied
-------------
1. _world_safe_llama_attention_forward: replaces LlamaAttention.forward (or
   Qwen2Attention.forward) with a version using only relative dims. GQA is handled
   by repeating k/v along the head axis using an expand+reshape that works on any
   leading batch/world dimensions. RoPE is applied via unsqueeze(-3) on cos/sin
   rather than unsqueeze(1), so it broadcasts correctly under world dims.

2. HookedLlama.forward: runs embedding and decoder layers directly, bypassing
   the model's forward method which manages caches and causal masks in ways that
   assume a fixed [batch, seq, d] shape.

Sites registered (identical naming to HookedGPT2/HookedGPTNeoX)
-----------------
    resid_pre_0   : embedding output
    head_{L}_{H}  : head H's contribution to the residual stream at layer L
    attn_out_{L}  : full attention output (sum of heads + bias)
    mlp_out_{L}   : MLP block output
    resid_post_{L}: residual after block L

Head decomposition uses nn.Linear convention for o_proj (weight=[d_model, d_model]):
    head_contribution = pre_proj[..., H*d_head:(H+1)*d_head] @ weight[:, H*d_head:(H+1)*d_head].T
where d_head = d_model // n_heads (query heads, since o_proj maps from n_heads*d_head).
The bias is applied once to the sum since it is a shared offset.

GQA note: the per-head decomposition iterates over query heads (n_heads), not
key-value heads. The output projection always maps from n_heads*d_head regardless
of GQA, so the decomposition is unaffected by the number of KV heads.
"""

import types
from typing import Optional

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model


def _repeat_kv_world_safe(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA without assuming a fixed number of leading dims.

    Standard repeat_kv unpacks shape as (batch, n_kv_heads, slen, d_head) which
    breaks under world dims. This version operates on the last 3 dims only.

    Args:
        hidden_states: [..., n_kv_heads, seq, d_head]
        n_rep: number of times to repeat each KV head (n_heads // n_kv_heads)

    Returns:
        [..., n_kv_heads * n_rep, seq, d_head]
    """
    if n_rep == 1:
        return hidden_states
    # Insert a rep dim: [..., n_kv_heads, 1, seq, d_head]
    hidden_states = hidden_states.unsqueeze(-3)
    # Expand along the rep dim: [..., n_kv_heads, n_rep, seq, d_head]
    expand_shape = (*hidden_states.shape[:-3], n_rep, *hidden_states.shape[-2:])
    hidden_states = hidden_states.expand(expand_shape)
    # Merge n_kv_heads and n_rep: [..., n_kv_heads*n_rep, seq, d_head]
    return hidden_states.reshape(
        *hidden_states.shape[:-4], -1, *hidden_states.shape[-2:]
    )


def _apply_rotary_pos_emb_world_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k tensors that may have leading world dimensions.

    Standard apply_rotary_pos_emb uses unsqueeze_dim=1 (absolute index at dim 1)
    to broadcast cos/sin over head dim. Under world dims q/k have shape
    [*world, batch, n_heads, seq, d_head] but cos/sin are [batch, seq, rotary_dim].
    We instead use unsqueeze(-3) to insert at the head axis relative to the end:
    cos becomes [batch, 1, seq, rotary_dim] which broadcasts over n_heads correctly
    regardless of how many world dims precede.

    Args:
        q, k : [..., n_heads, seq, d_head]
        cos, sin : [batch, seq, rotary_dim]

    Returns:
        Rotated q and k with the same shape.
    """
    cos = cos.unsqueeze(-3)  # [batch, 1, seq, rotary_dim]
    sin = sin.unsqueeze(-3)

    # Full-dim rotation (Llama uses full head_dim for RoPE)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half position.

    This is the same operation as transformers.models.llama.modeling_llama.rotate_half
    but inlined to avoid importing from a family-specific module.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _world_safe_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """LlamaAttention/Qwen2Attention forward replacement using relative dimension indices.

    The original uses .transpose(1, 2) to rearrange to head format and delegates to
    eager_attention_forward which uses absolute dims (.transpose(2, 3), .transpose(1, 2)).
    This replacement uses .transpose(-3, -2) throughout so it is safe under world dims.

    GQA: k/v are projected to n_kv_heads and repeated to n_heads via _repeat_kv_world_safe.
    RoPE: applied via _apply_rotary_pos_emb_world_safe (cos/sin unsqueeze at dim -3).

    Requires attn_implementation='eager'. Ignores past_key_values (no KV cache).
    """
    n_heads = self.config.num_attention_heads
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    n_kv_groups = n_heads // n_kv_heads

    # Project to query, key, value
    # [..., seq, d_model] → [..., seq, n_heads * d_head] → [..., seq, n_heads, d_head]
    q = self.q_proj(hidden_states).view(*hidden_states.shape[:-1], n_heads, head_dim)
    k = self.k_proj(hidden_states).view(*hidden_states.shape[:-1], n_kv_heads, head_dim)
    v = self.v_proj(hidden_states).view(*hidden_states.shape[:-1], n_kv_heads, head_dim)

    # [..., seq, n_heads, d_head] → [..., n_heads, seq, d_head]
    q = q.transpose(-3, -2)
    k = k.transpose(-3, -2)
    v = v.transpose(-3, -2)

    # Apply RoPE with world-safe unsqueeze
    cos, sin = position_embeddings
    q, k = _apply_rotary_pos_emb_world_safe(q, k, cos, sin)

    # Expand KV heads for GQA: [..., n_kv_heads, seq, d_head] → [..., n_heads, seq, d_head]
    k = _repeat_kv_world_safe(k, n_kv_groups)
    v = _repeat_kv_world_safe(v, n_kv_groups)

    # Scaled dot-product attention
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Context: [..., n_heads, seq, d_head]
    attn_output = torch.matmul(attn_weights, v)

    # Merge heads: [..., n_heads, seq, d_head] → [..., seq, n_heads * d_head]
    attn_output = (
        attn_output.transpose(-3, -2)
        .reshape(*attn_output.shape[:-3], attn_output.shape[-2], -1)
        .contiguous()
    )

    attn_output = self.o_proj(attn_output)
    return attn_output, None


class _LlamaAttentionHook:
    """Forward hook on layers[L].self_attn.o_proj (or self_attn for Qwen2).

    Decomposes the pre-o_proj output into per-head contributions using nn.Linear
    weight slices (weight=[d_model, n_heads*d_head], forward=x@W.T+b). Head H's
    contribution: pre_proj[..., H*d_head:(H+1)*d_head] @ weight[:, H*d_head:(H+1)*d_head].T
    The bias (if present) is added to the running sum as a shared offset.
    """

    def __init__(self, layer_idx: int, n_heads: int, d_head: int) -> None:
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        pre_proj = input[0]  # [*world, batch, seq, n_heads * d_head]
        weight = module.weight  # [d_model, n_heads*d_head] nn.Linear: x @ W.T
        bias = module.bias  # [d_model] or None

        attn_out = (
            bias.clone()
            if bias is not None
            else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        )
        for head_idx in range(self.n_heads):
            start = head_idx * self.d_head
            end = start + self.d_head
            head_contribution = pre_proj[..., start:end] @ weight[:, start:end].T
            head_contribution = pyro.deterministic(
                f"head_{self.layer_idx}_{head_idx}", head_contribution, event_dim=2
            )
            attn_out = attn_out + head_contribution
        return pyro.deterministic(f"attn_out_{self.layer_idx}", attn_out, event_dim=2)


class _LlamaMLPHook:
    """Forward hook on layers[L].mlp registering mlp_out_{L}."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        return pyro.deterministic(f"mlp_out_{self.layer_idx}", output, event_dim=2)


class HookedLlama(nn.Module):
    """Shared wrapper for Llama and Qwen2 model families.

    Both families have nearly identical attention and MLP structures, so a single
    class handles both. The model_type is detected from the config at construction
    and the appropriate HuggingFace model class is used.

    Supports:
    - LlamaForCausalLM / LlamaModel (Llama-3.1, Llama-2, etc.)
    - Qwen2ForCausalLM / Qwen2Model (Qwen2.5, Qwen2, etc.)

    Usage::

        model = HookedLlama("/path/to/llama-3.1-8B")
        trace = pyro.poutine.trace(model).get_trace(input_ids)

    Note: Llama-3.1-8B is 16 GB. For CI or CPU-only testing, use a synthetic
    LlamaConfig with small n_layer/n_head/hidden_size instead of real weights.
    """

    def __init__(
        self,
        model_name_or_config: str | LlamaConfig | Qwen2Config,
    ) -> None:
        super().__init__()

        if isinstance(model_name_or_config, (LlamaConfig, Qwen2Config)):
            config = model_name_or_config
            config._attn_implementation = "eager"
            self.model = _build_model_from_config(config)
        else:
            # Detect model family from saved config on disk
            disk_config = AutoConfig.from_pretrained(
                model_name_or_config, local_files_only=True
            )
            model_class = _select_model_class(disk_config.model_type)
            self.model = model_class.from_pretrained(
                model_name_or_config,
                attn_implementation="eager",
                local_files_only=True,
            )
            config = self.model.config

        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.d_model = config.hidden_size
        # head_dim may differ from d_model // n_heads for models with explicit head_dim
        self.d_head = getattr(config, "head_dim", None) or (
            self.d_model // self.n_heads
        )

        self._patch_attention_forwards()
        self._register_hooks()

    def _patch_attention_forwards(self) -> None:
        """Replace each attention block's forward with the world-safe version."""
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.forward = types.MethodType(_world_safe_llama_attention_forward, attn)

    def _register_hooks(self) -> None:
        """Register forward hooks for per-head contributions and MLP outputs."""
        self._hook_handles = []
        for layer_idx in range(self.n_layers):
            layer = self.model.layers[layer_idx]
            attn_hook = _LlamaAttentionHook(layer_idx, self.n_heads, self.d_head)
            mlp_hook = _LlamaMLPHook(layer_idx)
            self._hook_handles.append(
                layer.self_attn.o_proj.register_forward_hook(attn_hook)
            )
            self._hook_handles.append(layer.mlp.register_forward_hook(mlp_hook))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run Llama/Qwen2 and register residual stream sites.

        Runs embedding, rotary embedding, and decoder layers directly to control
        the residual stream. Avoids the model's own forward which uses caches and
        mask preparation that assume a fixed non-world-augmented shape.

        Args:
            input_ids: Long tensor of shape [batch, seq].

        Returns:
            hidden_states: Float tensor of shape [*world, batch, seq, d_model].
        """
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = pyro.deterministic("resid_pre_0", hidden_states, event_dim=2)

        # Rotary position embeddings: cos/sin each [batch, seq, d_head]
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Additive causal mask: [1, 1, seq, seq] with -inf above the diagonal
        causal_mask = torch.triu(
            torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=hidden_states.device
            ),
            diagonal=1,
        )

        for layer_idx, layer in enumerate(self.model.layers):
            # _LlamaAttentionHook and _LlamaMLPHook fire inside layer(), registering
            # head_{L}_{H}, attn_out_{L}, and mlp_out_{L}.
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            hidden_states = pyro.deterministic(
                f"resid_post_{layer_idx}", hidden_states, event_dim=2
            )

        return self.model.norm(hidden_states)


# ---------------------------------------------------------------------------
# Helpers for factory dispatch
# ---------------------------------------------------------------------------


def _select_model_class(model_type: str) -> type:
    """Return the HuggingFace model class for a given model_type string."""
    if model_type == "llama":
        return LlamaModel
    if model_type == "qwen2":
        return Qwen2Model
    raise ValueError(f"HookedLlama does not support model_type='{model_type}'")


def _build_model_from_config(config: LlamaConfig | Qwen2Config) -> nn.Module:
    """Instantiate the correct base model from a config object."""
    if isinstance(config, LlamaConfig):
        return LlamaModel(config)
    if isinstance(config, Qwen2Config):
        return Qwen2Model(config)
    raise TypeError(f"Unsupported config type: {type(config)}")
