"""Candidate ranking interface for prefiltering antecedent sites.

SearchForExplanation is memory-intensive (O(3^N) worlds for N antecedents),
so the full AC query should only run on a small ranked subset of candidate sites.
This module provides ranking functions to identify the most causally relevant
candidates before committing to the full AC inference.
"""

from typing import Callable, Optional

import torch
from chirho.interventional.handlers import do

from transformer.lib.cache import run_and_cache

RankingFn = Callable[[dict[str, torch.Tensor]], dict[str, float]]
# Takes site_name → activation cache, returns site_name → relevance score


def _activation_norm_ranking(cache: dict[str, torch.Tensor]) -> dict[str, float]:
    """Score each site by the Frobenius norm of its activation tensor.

    Norm is a cheap proxy for causal relevance: sites with near-zero activations
    contribute little to the residual stream and are unlikely to be causes.
    """
    return {name: activation.norm().item() for name, activation in cache.items()}


def rank_candidates(
    cache: dict[str, torch.Tensor],
    ranking_fn: Optional[RankingFn] = None,
    top_k: int = 10,
) -> list[str]:
    """Apply ranking_fn to cache and return top_k site names by descending score.

    Default ranking_fn: activation norm (cheap proxy for causal relevance).
    Ties are broken by site name for determinism.
    """
    fn = ranking_fn if ranking_fn is not None else _activation_norm_ranking
    scores = fn(cache)
    return sorted(scores, key=lambda s: (-scores[s], s))[:top_k]


def logit_diff_attribution(
    model_fn,
    input_args: tuple,
    sites: list[str],
    correct_token_id: int,
    incorrect_token_id: int,
    logit_site: str = "logits",
) -> dict[str, float]:
    """Compute direct logit difference attribution for each site.

    For each site S: attribution(S) = |logit_diff(clean) - logit_diff(zero_ablated_S)|.

    Uses zero ablation so no gradients or second forward pass over the full model
    are required — only one clean pass plus one ablated pass per candidate site.
    This is the patching-based attribution analogue of the causal tracing metric.

    Returns site_name → attribution score (higher = more causally relevant).
    """
    # Clean forward pass — cache all candidate sites and the logit site
    clean_cache = run_and_cache(model_fn, *input_args, sites=sites + [logit_site])
    clean_logits = clean_cache[logit_site]
    clean_diff = (
        (clean_logits[..., correct_token_id] - clean_logits[..., incorrect_token_id])
        .sum()
        .item()
    )

    attributions = {}
    for site in sites:
        if site not in clean_cache:
            attributions[site] = 0.0
            continue

        # Ablated pass: patch site to zero, run model, measure logit diff change
        zero_patch = torch.zeros_like(clean_cache[site])
        ablated_model = do(model_fn, {site: zero_patch})
        ablated_cache = run_and_cache(ablated_model, *input_args, sites=[logit_site])
        ablated_diff = (
            (
                ablated_cache[logit_site][..., correct_token_id]
                - ablated_cache[logit_site][..., incorrect_token_id]
            )
            .sum()
            .item()
        )

        attributions[site] = abs(clean_diff - ablated_diff)

    return attributions


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

    torch.manual_seed(42)
    _model = TinyTransformer(d_model=8, n_heads=2, n_layers=2)
    _model.eval()
    x_in = torch.randn(1, 5, 8)

    # Add a logit site so logit_diff_attribution has something to measure
    def toy_model_fn(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        out = _model(inp)
        return pyro.deterministic("logits", out, event_dim=2)

    all_sites = ["head_0_0", "head_0_1", "head_1_0", "head_1_1", "resid_0", "resid_1"]

    print("=== prefilter.py validation ===")

    # rank_candidates with default norm ranking
    cache = run_and_cache(toy_model_fn, x_in, sites=all_sites)
    ranked = rank_candidates(cache, top_k=3)
    assert len(ranked) == 3, f"Expected 3 candidates, got {len(ranked)}"
    assert all(s in all_sites for s in ranked), "Ranked sites not in all_sites"
    print(f"Top-3 by activation norm: {ranked}")

    # Custom ranking function (inverse norm — least active first)
    def inverse_norm(c: dict[str, torch.Tensor]) -> dict[str, float]:
        return {name: -act.norm().item() for name, act in c.items()}

    inv_ranked = rank_candidates(cache, ranking_fn=inverse_norm, top_k=3)
    assert ranked != inv_ranked, "Custom ranking should differ from default"
    print(f"Top-3 by inverse norm (lowest activity): {inv_ranked}")

    # logit_diff_attribution — correct=0, incorrect=1 (arbitrary for toy model)
    attributions = logit_diff_attribution(
        model_fn=toy_model_fn,
        input_args=(x_in,),
        sites=all_sites,
        correct_token_id=0,
        incorrect_token_id=1,
        logit_site="logits",
    )
    assert set(attributions.keys()) == set(all_sites), "Attribution keys mismatch"
    assert all(v >= 0.0 for v in attributions.values()), (
        "Attributions should be non-negative"
    )

    ranked_by_attribution = sorted(attributions, key=lambda s: -attributions[s])
    print(f"Sites ranked by logit diff attribution: {ranked_by_attribution}")
    print(
        f"Top attribution: {ranked_by_attribution[0]} = {attributions[ranked_by_attribution[0]]:.4f}"
    )

    # Use logit_diff_attribution as a ranking_fn via rank_candidates
    def attr_ranking(c: dict[str, torch.Tensor]) -> dict[str, float]:
        # c here is the cache; recompute attributions for the sites in c
        return logit_diff_attribution(
            model_fn=toy_model_fn,
            input_args=(x_in,),
            sites=list(c.keys()),
            correct_token_id=0,
            incorrect_token_id=1,
            logit_site="logits",
        )

    attr_top = rank_candidates(cache, ranking_fn=attr_ranking, top_k=2)
    print(f"Top-2 by attribution ranking_fn: {attr_top}")
    print(
        "PASS: prefilter.py rank_candidates and logit_diff_attribution work correctly"
    )
