"""Activation caching via pyro.poutine.trace.

Captures pyro.deterministic sites registered during a model forward pass.
Equivalent to TransformerLens run_with_cache for Pyro-wrapped models.
"""

from typing import Optional

import pyro
import torch


def run_and_cache(
    model_fn,
    *args,
    sites: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """Run model_fn and return a dict mapping site name to activation tensor.

    Captures every pyro.deterministic site registered during the forward pass.
    sites filters to specific names; None returns all deterministic sites.

    model_fn must be a callable that calls pyro.deterministic internally.
    """
    trace = pyro.poutine.trace(model_fn).get_trace(*args)

    # pyro.deterministic registers as sample sites with Delta distributions;
    # type == "sample" captures them while excluding internal Pyro bookkeeping.
    cache = {
        name: node["value"]
        for name, node in trace.nodes.items()
        if node["type"] == "sample"
    }

    if sites is not None:
        cache = {name: cache[name] for name in sites if name in cache}

    return cache


if __name__ == "__main__":
    import torch.nn as nn

    # ---------------------------------------------------------------------------
    # Toy transformer from q2_transformer_wrapper.py
    # ---------------------------------------------------------------------------

    class AttentionHead(nn.Module):
        def __init__(self, d_model, d_head):
            super().__init__()
            self.W_Q = nn.Linear(d_model, d_head, bias=False)
            self.W_K = nn.Linear(d_model, d_head, bias=False)
            self.W_V = nn.Linear(d_model, d_head, bias=False)
            self.scale = d_head**-0.5

        def forward(self, x):
            q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
            return torch.softmax(q @ k.mT * self.scale, dim=-1) @ v

    class AttentionLayer(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.d_head = d_model // n_heads
            self.heads = nn.ModuleList(
                [AttentionHead(d_model, d_model // n_heads) for _ in range(n_heads)]
            )
            self.W_O = nn.Linear(d_model // n_heads, d_model, bias=False)

        def forward(self, x, layer_idx):
            resid = x
            for h, head in enumerate(self.heads):
                h_out = head(x)
                h_out = pyro.deterministic(f"head_{layer_idx}_{h}", h_out, event_dim=2)
                resid = resid + self.W_O(h_out)
            return resid

    class TinyTransformer(nn.Module):
        def __init__(self, d_model=8, n_heads=2, n_layers=2):
            super().__init__()
            self.layers = nn.ModuleList(
                [AttentionLayer(d_model, n_heads) for _ in range(n_layers)]
            )

        def forward(self, x):
            resid = x
            for layer_idx, layer in enumerate(self.layers):
                resid = layer(resid, layer_idx)
                resid = pyro.deterministic(f"resid_{layer_idx}", resid, event_dim=2)
            return resid

    torch.manual_seed(42)
    _model = TinyTransformer(d_model=8, n_heads=2, n_layers=2)
    _model.eval()
    x_in = torch.randn(1, 5, 8)

    def toy_model_fn(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        return _model(inp)

    print("=== cache.py validation ===")

    # All sites
    full_cache = run_and_cache(toy_model_fn, x_in)
    print(f"All sites: {sorted(full_cache.keys())}")
    assert "head_0_0" in full_cache, "head_0_0 should be cached"
    assert "resid_1" in full_cache, "resid_1 should be cached"

    # Filtered sites
    filtered = run_and_cache(toy_model_fn, x_in, sites=["head_0_0", "resid_0"])
    assert set(filtered.keys()) == {"head_0_0", "resid_0"}, "Site filter failed"
    print(f"Filtered sites: {sorted(filtered.keys())}")
    print(f"head_0_0 shape: {filtered['head_0_0'].shape}")

    # Sanity: same input gives same cached values
    cache_a = run_and_cache(toy_model_fn, x_in)
    cache_b = run_and_cache(toy_model_fn, x_in)
    assert torch.allclose(cache_a["head_0_0"], cache_b["head_0_0"]), (
        "Cache not deterministic"
    )
    print("PASS: run_and_cache is deterministic and filters correctly")
