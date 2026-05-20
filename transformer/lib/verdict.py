"""Posterior extraction from SearchForExplanation traces.

Aggregates case variable distributions across Monte Carlo samples to produce
PNS, necessity, and sufficiency estimates per antecedent site.

Case variable semantics (from chirho.explainable):
  0 = factual world: no intervention applied
  1 = necessity world: antecedent replaced with alternative
  2 = sufficiency world: antecedent restored to observed value
"""

import torch


def extract_case_variables(
    traces: list[dict],
    prefix: str = "__cause____antecedent_",
) -> dict[str, torch.Tensor]:
    """Extract case variable values for each antecedent site across traces.

    Returns dict: site_name → 1D tensor of case values (one per trace).
    Traces that lack a given case variable are silently skipped for that site.
    """
    # Collect site names from the first trace that has case variables
    case_keys = [key for key in (traces[0] if traces else {}) if key.startswith(prefix)]

    result = {}
    for key in case_keys:
        # Strip prefix to get the original site name
        site_name = key[len(prefix) :]
        values = [tr[key].item() for tr in traces if key in tr]
        result[site_name] = torch.tensor(values, dtype=torch.long)

    return result


def compute_pns(case_values: torch.Tensor) -> float:
    """P(case != 0) — probability that the site is an actual cause.

    PNS combines necessity (case=1) and sufficiency (case=2): a site is a
    cause in any world where the posterior selects a non-factual case.
    """
    return (case_values != 0).float().mean().item()


def compute_necessity(case_values: torch.Tensor) -> float:
    """P(case == 1) — probability of the necessity world being selected.

    Necessity: the antecedent's absence (alternative value) changes the consequent.
    """
    return (case_values == 1).float().mean().item()


def compute_sufficiency(case_values: torch.Tensor) -> float:
    """P(case == 2) — probability of the sufficiency world being selected.

    Sufficiency: the antecedent's presence (observed value) is enough to
    produce the consequent even when other sites take alternative values.
    """
    return (case_values == 2).float().mean().item()


def summarize_verdict(
    traces: list[dict],
    antecedent_sites: list[str],
    prefix: str = "__cause____antecedent_",
) -> dict[str, dict]:
    """Return per-site verdict dict with keys: pns, necessity, sufficiency, n_samples.

    antecedent_sites filters to only the requested sites.
    Sites absent from the traces produce an empty entry with n_samples=0.
    """
    case_variables = extract_case_variables(traces, prefix=prefix)

    return {
        site: (
            {
                "pns": compute_pns(case_variables[site]),
                "necessity": compute_necessity(case_variables[site]),
                "sufficiency": compute_sufficiency(case_variables[site]),
                "n_samples": len(case_variables[site]),
            }
            if site in case_variables
            else {
                "pns": float("nan"),
                "necessity": float("nan"),
                "sufficiency": float("nan"),
                "n_samples": 0,
            }
        )
        for site in antecedent_sites
    }
