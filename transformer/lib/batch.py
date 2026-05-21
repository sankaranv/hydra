"""Batch AC query runner for aggregating PNS across multiple prompts.

Mechanistic interpretability often requires asking "is this site consistently
causal across many inputs?" rather than for a single input. batch_ac_queries
runs train_ac_guide for each prompt spec independently and collects the per-site
PNS values, giving a distribution the caller can aggregate (mean, median, etc.).
"""

from transformer.lib.ac_query import train_ac_guide


def batch_ac_queries(
    prompt_specs: list[dict],
    n_steps: int = 300,
    event_dim: int = 2,
    consequent_scale: float = 0.1,
    first_available_dim: int = -8,
) -> dict[str, list[float]]:
    """Run train_ac_guide for each prompt and collect per-site PNS values.

    Each element of prompt_specs is a dict with keys:
      model_fn         Callable — zero-argument closure over the prompt's input
      antecedents      dict[str, Tensor] — site → observed activation value
      alternatives     dict[str, Tensor] — site → alternative (e.g. zero-ablated)
      consequent_site  str — name of the logit/output site
      consequent_value Tensor — observed value of the consequent site
      witnesses        dict[str, None] | None (optional, defaults to {})

    Returns dict[site_name → list[float]] where each list has one PNS entry per
    prompt. Sites absent from a prompt's antecedents produce nan for that prompt.
    Use to assess how consistently a site is causal across a prompt distribution:
      mean_pns = {site: sum(v for v in vals if v == v) / len(vals)
                  for site, vals in result.items()}
    """
    all_sites: set[str] = set()
    for spec in prompt_specs:
        all_sites.update(spec["antecedents"].keys())

    site_pns: dict[str, list[float]] = {site: [] for site in all_sites}

    for spec in prompt_specs:
        _, pns_dict = train_ac_guide(
            spec["model_fn"],
            antecedents=spec["antecedents"],
            alternatives=spec["alternatives"],
            witnesses=spec.get("witnesses", {}),
            consequent_site=spec["consequent_site"],
            consequent_value=spec["consequent_value"],
            event_dim=event_dim,
            consequent_scale=consequent_scale,
            first_available_dim=first_available_dim,
            n_steps=n_steps,
        )
        for site in all_sites:
            site_pns[site].append(pns_dict.get(site, float("nan")))

    return site_pns
