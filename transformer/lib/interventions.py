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

    Precondition: all sites must use the same format (either all single-reference
    or all stacked). Mixing formats breaks the shared-index invariant: a
    single-reference site always returns the same value regardless of shared_idx,
    so cross-site consistency only holds when all sites are stacked.
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
