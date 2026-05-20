"""
GPT-2 wrapper with pyro.deterministic sites for ChiRho interventions.

Architecture overview
---------------------
GPT-2 uses Conv1D internally and its original attention forward relies on
.split(dim=2) and .transpose(1, 2), which hardcode absolute tensor dimensions.
When ChiRho's MultiWorldCounterfactual injects leading world dimensions these
calls break. Additionally, GPT2Model.forward calls hidden_states.view(-1, ...) at
the end, which collapses world and batch dimensions.

Fixes applied
-------------
1. world_safe_attention_forward: replaces GPT2Attention.forward with an equivalent
   that uses only relative dimensions (dim=-1, transpose(-3, -2)) and avoids
   calling eager_attention_forward (whose internal transpose(1, 2) is also absolute).
   All models are loaded with attn_implementation='eager' to ensure a single code path.

2. HookedGPT2.forward: runs the transformer step-by-step instead of delegating to
   GPT2Model.forward, skipping its final hidden_states.view(-1, seq, d) which would
   merge world and batch dimensions.

Sites registered
----------------
For each layer L in range(n_layers):
    head_{L}_{H}  : per-head contribution to residual stream, shape [*batch, seq, d_model]
                    computed as pre_proj[..., H*d_head:(H+1)*d_head] @ W_proj[H*d_head:, :]
                    where W_proj is c_proj.weight ([d_model, d_model] Conv1D convention).
    attn_out_{L}  : full attention output = sum of head contributions + c_proj bias,
                    shape [*batch, seq, d_model].
    mlp_out_{L}   : MLP output, shape [*batch, seq, d_model].
    resid_post_{L}: residual stream after block L = resid_pre_L + attn_out_L + mlp_out_L,
                    shape [*batch, seq, d_model].

resid_pre_0 = embedding (token + position), registered as resid_post_{-1} is not exposed.
resid_pre_L for L > 0 is identical to resid_post_{L-1}.

Design choice: head sites have shape [*batch, seq, d_model], not [*batch, seq, d_head].
Each head H produces a full-rank contribution via c_proj weight slice [d_head, d_model];
registering the d_model-shaped result means patching head_L_H is a semantically clean
intervention that replaces head H's entire causal contribution to the residual stream.
event_dim=2 throughout because [seq, d_model] is the natural event tensor.
"""

import types
from typing import Optional

import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


# ---------------------------------------------------------------------------
# World-safe attention forward
# ---------------------------------------------------------------------------


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
      does its own .transpose(1, 2) that would break world-indexed tensors)

    This method is monkey-patched onto each GPT2Attention block at construction.
    Requires attn_implementation='eager' in the model config.
    Requires use_cache=False (past_key_values is ignored).
    """
    q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=-1)

    # [*batch, seq, d_model] -> [*batch, seq, n_heads, d_head] -> [*batch, n_heads, seq, d_head]
    shape = (*q.shape[:-1], -1, self.head_dim)
    q = q.view(shape).transpose(-3, -2)
    k = k.view(shape).transpose(-3, -2)
    v = v.view(shape).transpose(-3, -2)

    # Scaled dot-product attention: [*batch, n_heads, seq, seq]
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Context: [*batch, n_heads, seq, d_head]
    attn_output = torch.matmul(attn_weights, v)

    # Merge heads: [*batch, n_heads, seq, d_head] -> [*batch, seq, d_model]
    # shape[:-3] = leading batch dims; shape[-2] = seq (second-to-last of original)
    attn_output = (
        attn_output.transpose(-3, -2)
        .reshape(*attn_output.shape[:-3], attn_output.shape[-2], -1)
        .contiguous()
    )

    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Per-head contribution hook on c_proj
# ---------------------------------------------------------------------------


class _AttentionHook:
    """Forward hook on model.h[L].attn.c_proj.

    Receives the pre-c_proj merged attention output ('pre_proj'), decomposes it
    into per-head contributions using c_proj weight slices, registers each as a
    pyro.deterministic site, and returns a reconstructed attn_out from the
    (possibly patched) head contributions.

    Conv1D convention: c_proj.weight has shape [d_model, d_model], and the
    forward computes x @ weight + bias. Head H's contribution is:
        pre_proj[..., H*d_head:(H+1)*d_head] @ weight[H*d_head:(H+1)*d_head, :]
    which gives a [*batch, seq, d_model] tensor — the full-rank contribution of
    head H to the residual stream.

    The c_proj bias is added to the sum of head contributions (not per-head)
    because the bias is a shared offset, not head-specific.
    """

    def __init__(self, layer_idx: int, n_heads: int, d_head: int) -> None:
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.d_head = d_head

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        pre_proj = input[0]  # [*batch, seq, d_model]
        weight = module.weight  # [d_model, d_model] Conv1D: x @ weight
        bias = module.bias  # [d_model]

        # Accumulate starting from bias to match the native c_proj order:
        # bias + head_0 + head_1 + ... ensures float32 rounding is identical to
        # what a forward hook that starts from bias.clone() and adds heads would produce.
        attn_out = bias.clone()
        for head_idx in range(self.n_heads):
            start = head_idx * self.d_head
            end = start + self.d_head
            # Project this head's d_head-dim slice into d_model space
            head_contribution = pre_proj[..., start:end] @ weight[start:end, :]
            head_contribution = pyro.deterministic(
                f"head_{self.layer_idx}_{head_idx}", head_contribution, event_dim=2
            )
            attn_out = attn_out + head_contribution
        attn_out = pyro.deterministic(
            f"attn_out_{self.layer_idx}", attn_out, event_dim=2
        )
        return attn_out


class _MLPHook:
    """Forward hook on model.h[L].mlp.

    Registers the MLP output as a pyro.deterministic site and returns the
    (possibly patched) value for downstream use.
    """

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def __call__(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        return pyro.deterministic(f"mlp_out_{self.layer_idx}", output, event_dim=2)


# ---------------------------------------------------------------------------
# HookedGPT2
# ---------------------------------------------------------------------------


class HookedGPT2(nn.Module):
    """GPT-2 wrapper with pyro.deterministic sites for ChiRho interventions.

    Loads a GPT-2 model (or uses a provided GPT2Config for a tiny model) and
    registers intermediate activations as named sites. All sites use event_dim=2
    since the natural event tensor is [seq, d_model].

    Loaded with attn_implementation='eager' to ensure a deterministic, inspectable
    code path through the attention mechanism. The KV cache is disabled (use_cache=False)
    in forward() because it is not compatible with world-safe attention.

    Usage::

        model = HookedGPT2("gpt2")
        trace = pyro.poutine.trace(model).get_trace(input_ids)

        # Counterfactual intervention
        with MultiWorldCounterfactual(first_available_dim=-9):
            with do(actions={"head_0_0": patch}):
                out = model(input_ids)
            factual = gather(out, IndexSet(head_0_0={0}), event_dim=2)
            intervened = gather(out, IndexSet(head_0_0={1}), event_dim=2)
    """

    def __init__(self, model_name_or_config: str | GPT2Config = "gpt2") -> None:
        super().__init__()

        if isinstance(model_name_or_config, GPT2Config):
            config = model_name_or_config
            # Force eager attention so world_safe_attention_forward is used
            config._attn_implementation = "eager"
            self.gpt2 = GPT2Model(config)
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                model_name_or_config,
                attn_implementation="eager",
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
        which would merge world and batch dimensions when running under
        MultiWorldCounterfactual. Instead, runs the embedding and transformer
        blocks directly with an explicit causal mask.

        Args:
            input_ids: Long tensor of shape [batch, seq].

        Returns:
            hidden_states: Float tensor of shape [*world_dims, batch, seq, d_model].
        """
        # Token + position embeddings -> [batch, seq, d_model]
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = self.gpt2.drop(inputs_embeds + position_embeds)

        # Register embedding as resid_pre_0 (initial residual stream)
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
            # head_{L}_{H}, attn_out_{L}, and mlp_out_{L} as pyro.deterministic sites.
            hidden_states = block(
                hidden_states,
                past_key_values=None,
                attention_mask=causal_mask,
                use_cache=False,
            )
            # Register post-block residual stream; pyro.deterministic return value
            # must be used downstream so patching resid_post_L propagates.
            hidden_states = pyro.deterministic(
                f"resid_post_{layer_idx}", hidden_states, event_dim=2
            )

        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Site name utility
# ---------------------------------------------------------------------------


def site_names(n_layers: int, n_heads: int) -> list[str]:
    """Return the full list of pyro.deterministic site names for a GPT-2 model.

    Ordered as they are registered during a forward pass: embedding first,
    then for each layer the head contributions, attention output, MLP output,
    and residual stream.

    Args:
        n_layers: Number of transformer layers (e.g. 12 for GPT-2 small).
        n_heads: Number of attention heads (e.g. 12 for GPT-2 small).

    Returns:
        List of site name strings.
    """
    names = ["resid_pre_0"]
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            names.append(f"head_{layer_idx}_{head_idx}")
        names.append(f"attn_out_{layer_idx}")
        names.append(f"mlp_out_{layer_idx}")
        names.append(f"resid_post_{layer_idx}")
    return names


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------


def verify_wrapper_correctness(model_name: str = "gpt2") -> bool:
    """Verify that do() patching matches native hook injection.

    Loads (or constructs) a GPT-2 model, patches head_0_0 to zeros via two methods:
    (1) ChiRho's do() handler intercepting the pyro.deterministic site, and
    (2) a native PyTorch forward hook that zeroes head 0's c_proj contribution.

    Asserts the two outputs agree to within 1e-5. Prints a summary.

    Args:
        model_name: HuggingFace model name string, or "tiny" to use a 2-layer
                    2-head d_model=8 model (avoids network access for testing).

    Returns:
        True if verification passes, False otherwise.
    """
    from chirho.interventional.handlers import do

    if model_name == "tiny":
        config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=8,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
        wrapper = HookedGPT2(config)
    else:
        try:
            wrapper = HookedGPT2(model_name)
        except OSError:
            print(
                f"  Network unavailable for {model_name!r}; falling back to tiny model."
            )
            config = GPT2Config(
                n_layer=2,
                n_head=2,
                n_embd=8,
                attn_pdrop=0.0,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
            )
            wrapper = HookedGPT2(config)

    wrapper.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, wrapper.gpt2.config.vocab_size, (1, 5))

    # Determine patch shape: head sites are [batch, seq, d_model]
    batch, seq = input_ids.shape
    d_model = wrapper.d_model
    patch = torch.zeros(batch, seq, d_model)

    # --- Method 1: ChiRho do() patching ---
    with torch.no_grad():
        out_chirho = do(wrapper, {"head_0_0": patch})(input_ids)

    # --- Method 2: native hook injecting zeros for head 0's contribution ---
    # The hook reconstructs c_proj output with head 0's slice zeroed out.
    d_head = wrapper.d_head
    attn_module = wrapper.gpt2.h[0].attn

    def _zero_head_0_hook(module, input, output):  # noqa: ARG001
        pre_proj = input[0]
        weight = module.weight
        bias = module.bias
        # Reconstruct without head 0's contribution (set to zero)
        result = bias.clone()
        for head_idx in range(wrapper.n_heads):
            start, end = head_idx * d_head, (head_idx + 1) * d_head
            if head_idx == 0:
                continue  # zero ablation of head 0
            result = result + pre_proj[..., start:end] @ weight[start:end, :]
        return result

    hook_handle = attn_module.c_proj.register_forward_hook(_zero_head_0_hook)
    with torch.no_grad():
        out_hook = wrapper(input_ids)
    hook_handle.remove()

    max_diff = (out_chirho - out_hook).abs().max().item()
    passed = max_diff < 1e-5

    print("=" * 60)
    print("verify_wrapper_correctness")
    print("=" * 60)
    print(
        f"  Model: {model_name}  (n_layers={wrapper.n_layers}, n_heads={wrapper.n_heads}, "
        f"d_model={wrapper.d_model})"
    )
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Patch: head_0_0 -> zeros  ({patch.shape})")
    print(f"  ChiRho do() output norm:  {out_chirho.norm().item():.6f}")
    print(f"  Native hook output norm:  {out_hook.norm().item():.6f}")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    if passed:
        print("  PASS: do() patching matches native hook injection")
    else:
        print("  FAIL: outputs differ — max diff exceeds 1e-5")
    return passed


# ---------------------------------------------------------------------------
# Script entry point for smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
    from chirho.indexed.ops import IndexSet, gather
    from chirho.interventional.handlers import do

    print("=" * 60)
    print("HookedGPT2 smoke test")
    print("=" * 60)

    # Use a tiny config to avoid network dependency
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=8,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = HookedGPT2(config)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    seq_len = input_ids.shape[-1]
    d_model = model.d_model

    # --- Test 1: sites in trace ---
    print("\nTest 1: pyro.poutine.trace captures all sites")
    trace = pyro.poutine.trace(model).get_trace(input_ids)
    registered = {
        name: node["value"].shape
        for name, node in trace.nodes.items()
        if node["type"] == "sample"
    }
    expected = site_names(model.n_layers, model.n_heads)
    print(f"  Expected {len(expected)} sites, got {len(registered)}")
    for name in expected:
        if name in registered:
            print(f"    {name}: {tuple(registered[name])}")
        else:
            print(f"    {name}: MISSING")
    all_present = all(name in registered for name in expected)
    print(f"  {'PASS' if all_present else 'FAIL'}: all expected sites present")

    # --- Test 2: do() patching ---
    print("\nTest 2: do() patching")
    patch = torch.zeros(1, seq_len, d_model)
    with torch.no_grad():
        out_clean = model(input_ids)
        out_patched = do(model, {"head_0_0": patch})(input_ids)
    print(f"  Clean output norm:   {out_clean.norm().item():.4f}")
    print(f"  Patched output norm: {out_patched.norm().item():.4f}")
    print(f"  Outputs differ: {not torch.allclose(out_clean, out_patched, atol=1e-4)}")

    # --- Test 3: MultiWorldCounterfactual ---
    print("\nTest 3: MultiWorldCounterfactual")
    # first_available_dim must be negative enough for event_dim=2 sites plus world dims
    with MultiWorldCounterfactual(first_available_dim=-9):
        with do(actions={"head_0_0": patch}):
            out_mwc = model(input_ids)
        out_f = gather(out_mwc, IndexSet(**{"head_0_0": {0}}), event_dim=2)
        out_i = gather(out_mwc, IndexSet(**{"head_0_0": {1}}), event_dim=2)

    factual_matches = torch.allclose(out_f.squeeze(0), out_clean, atol=1e-4)
    intervened_matches = torch.allclose(out_i.squeeze(0), out_patched, atol=1e-4)
    print(f"  MWC output shape: {out_mwc.shape}")
    print(f"  Factual world norm: {out_f.norm().item():.4f}")
    print(f"  Intervened world norm: {out_i.norm().item():.4f}")
    print(f"  Factual matches clean run: {factual_matches}")
    print(f"  Intervened matches do() run: {intervened_matches}")
    print(
        f"  {'PASS' if factual_matches and intervened_matches else 'FAIL'}: MWC worlds correct"
    )

    # --- Test 4: correctness verification ---
    print()
    passed = verify_wrapper_correctness("tiny")
    print()
    print("=" * 60)
    print(
        f"Overall: {'PASS' if passed and all_present and factual_matches and intervened_matches else 'FAIL'}"
    )
    print("=" * 60)
