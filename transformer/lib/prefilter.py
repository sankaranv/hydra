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

    Uses zero ablation so no gradients are required. Cost: one clean pass plus
    one ablated pass per candidate site (not two total — one ablated pass per site).
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


def prefilter_candidates(
    model_fn: Callable,
    input_args: tuple,
    candidate_sites: list[str],
    top_k: int = 10,
    correct_token_id: Optional[int] = None,
    incorrect_token_id: Optional[int] = None,
    logit_site: str = "logits",
    ranking_fn: Optional[RankingFn] = None,
) -> list[str]:
    """Return the top_k most causally relevant sites from candidate_sites.

    This is the main entry point for the pre-filter stage. Run before
    train_ac_guide to restrict the expensive AC inference to a small set
    of high-priority candidates.

    Ranking priority:
      1. ranking_fn (if provided) — applied to the activation cache
      2. logit_diff_attribution (if correct_token_id and incorrect_token_id
         are provided) — one zero-ablation pass per candidate site, no gradients
      3. activation norm (fallback) — single forward pass, O(1) per site

    Logit difference attribution is the recommended default: it measures how
    much zero-ablating each site changes the logit_diff(correct, incorrect),
    which is a linear approximation of causal influence on the output.
    """
    if ranking_fn is not None:
        cache = run_and_cache(model_fn, *input_args, sites=candidate_sites)
        return rank_candidates(cache, ranking_fn=ranking_fn, top_k=top_k)

    if correct_token_id is not None and incorrect_token_id is not None:
        scores = logit_diff_attribution(
            model_fn,
            input_args,
            candidate_sites,
            correct_token_id=correct_token_id,
            incorrect_token_id=incorrect_token_id,
            logit_site=logit_site,
        )
        return sorted(scores, key=lambda s: (-scores[s], s))[:top_k]

    # Activation norm fallback: single forward pass, no token IDs required.
    cache = run_and_cache(model_fn, *input_args, sites=candidate_sites)
    return rank_candidates(cache, top_k=top_k)
