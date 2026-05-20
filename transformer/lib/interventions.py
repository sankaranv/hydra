"""Alternative value construction for SearchForExplanation antecedents.

These functions produce pre-specified alternative tensors that bypass
SearchForExplanation's default Cauchy proposal, which draws extreme values
outside the range of transformer activations (see q4_search_for_explanation.py).

All ablations operate on activation caches: dicts of site_name → tensor.
"""

import torch

from transformer.lib.cache import run_and_cache


def zero_ablation(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return dict mapping site to a zero tensor of the same shape.

    Appropriate for necessity interventions when the null baseline is meaningful,
    e.g., "what would happen if this head contributed nothing?"
    """
    return {name: torch.zeros_like(activation) for name, activation in cache.items()}


def mean_ablation(
    model_fn,
    inputs: list,
    sites: list[str],
) -> dict[str, torch.Tensor]:
    """Run model_fn on each input and average activations per site.

    inputs is a list of positional arg tuples, one per prompt.
    Each element should be a tuple: (arg1, arg2, ...) passed as model_fn(*args).
    For single-arg models pass [(x1,), (x2,), ...].

    Returns site_name → mean activation across all inputs.
    """
    # Accumulate per-site tensors across all inputs
    accumulated: dict[str, list[torch.Tensor]] = {site: [] for site in sites}
    for input_args in inputs:
        # Support both bare tensors and tuples of args
        args = input_args if isinstance(input_args, tuple) else (input_args,)
        cache = run_and_cache(model_fn, *args, sites=sites)
        for site in sites:
            if site in cache:
                accumulated[site].append(cache[site])

    return {
        site: torch.stack(tensors).mean(dim=0)
        for site, tensors in accumulated.items()
        if tensors
    }


def resample_ablation(
    cache: dict[str, torch.Tensor],
    reference_cache: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """For each site, draw one random sample from reference_cache.

    reference_cache should be a dict of site_name → stacked tensor with shape
    [n_prompts, *activation_shape], where dim 0 indexes different prompts.
    A single random prompt index is drawn and used for all sites to maintain
    cross-site consistency within the resampled alternative.

    If reference_cache[site] is 1D along dim 0 (a single reference activation
    with the same shape as cache[site]), it is returned directly.
    """
    # Pick one prompt index uniformly; use the same index for all sites so the
    # alternative is a coherent single-prompt activation profile.
    result = {}
    shared_idx: int | None = None

    for site, activation in cache.items():
        if site not in reference_cache:
            continue
        ref = reference_cache[site]
        # If ref has the same shape as activation, use it directly (single reference)
        if ref.shape == activation.shape:
            result[site] = ref
        else:
            # ref is stacked: [n_prompts, *activation_shape]
            n_prompts = ref.shape[0]
            if shared_idx is None:
                shared_idx = int(torch.randint(n_prompts, (1,)).item())
            result[site] = ref[shared_idx]

    return result


def build_alternatives(
    sites: list[str],
    method: str,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Convenience wrapper dispatching to zero, mean, or resample ablation.

    method: one of "zero", "mean", "resample"

    For "zero": pass cache=<site→tensor dict>
    For "mean": pass model_fn=<callable>, inputs=<list of arg tuples>, (sites inferred)
    For "resample": pass cache=<site→tensor dict>, reference_cache=<site→tensor dict>
    """
    if method == "zero":
        cache = kwargs["cache"]
        site_cache = {s: cache[s] for s in sites if s in cache}
        return zero_ablation(site_cache)

    elif method == "mean":
        return mean_ablation(
            model_fn=kwargs["model_fn"],
            inputs=kwargs["inputs"],
            sites=sites,
        )

    elif method == "resample":
        cache = kwargs["cache"]
        reference_cache = kwargs["reference_cache"]
        site_cache = {s: cache[s] for s in sites if s in cache}
        return resample_ablation(site_cache, reference_cache)

    else:
        raise ValueError(
            f"Unknown ablation method '{method}'. Choose: zero, mean, resample"
        )


if __name__ == "__main__":
    import pyro
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

    def toy_model_fn(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        return TinyTransformer()(inp)

    torch.manual_seed(42)
    x_in = torch.randn(1, 5, 8)
    x_alt = torch.randn(1, 5, 8)
    x_alt2 = torch.randn(1, 5, 8)

    print("=== interventions.py validation ===")

    # Baseline cache
    cache = run_and_cache(toy_model_fn, x_in, sites=["head_0_0", "head_0_1"])
    print(
        f"Cache sites: {sorted(cache.keys())}, head_0_0 shape: {cache['head_0_0'].shape}"
    )

    # zero_ablation
    zeros = zero_ablation(cache)
    assert all(v.sum() == 0.0 for v in zeros.values()), (
        "zero_ablation should give all zeros"
    )
    print("PASS: zero_ablation returns zero tensors of correct shape")

    # mean_ablation
    means = mean_ablation(
        toy_model_fn,
        inputs=[(x_in,), (x_alt,), (x_alt2,)],
        sites=["head_0_0"],
    )
    assert means["head_0_0"].shape == cache["head_0_0"].shape, "mean shape mismatch"
    print(f"PASS: mean_ablation returns correct shape {means['head_0_0'].shape}")

    # resample_ablation — stack two references along dim 0
    ref_cache_0 = run_and_cache(toy_model_fn, x_alt, sites=["head_0_0", "head_0_1"])
    ref_cache_1 = run_and_cache(toy_model_fn, x_alt2, sites=["head_0_0", "head_0_1"])
    stacked_ref = {
        site: torch.stack([ref_cache_0[site], ref_cache_1[site]])
        for site in ["head_0_0", "head_0_1"]
    }
    resampled = resample_ablation(cache, stacked_ref)
    assert resampled["head_0_0"].shape == cache["head_0_0"].shape, (
        "resample shape mismatch"
    )
    print(
        f"PASS: resample_ablation returns correct shape {resampled['head_0_0'].shape}"
    )

    # build_alternatives dispatch
    alts_zero = build_alternatives(["head_0_0"], method="zero", cache=cache)
    assert alts_zero["head_0_0"].sum() == 0.0
    print("PASS: build_alternatives dispatches correctly")
