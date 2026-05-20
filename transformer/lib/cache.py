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
