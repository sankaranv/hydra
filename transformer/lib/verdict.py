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


if __name__ == "__main__":
    import pyro
    import torch.nn as nn

    from transformer.lib.ac_query import run_ac_query
    from transformer.lib.cache import run_and_cache
    from transformer.lib.interventions import zero_ablation

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

    def toy_model_fn():
        inp = pyro.deterministic("input", x_in, event_dim=2)
        out = _model(inp)
        return pyro.deterministic("logits", out, event_dim=2)

    print("=== verdict.py validation ===")

    obs_cache = run_and_cache(toy_model_fn, sites=["head_0_0", "logits"])
    alt_h00 = zero_ablation({"head_0_0": obs_cache["head_0_0"]})

    traces = run_ac_query(
        model_fn=toy_model_fn,
        antecedents={"head_0_0": obs_cache["head_0_0"]},
        alternatives=alt_h00,
        witnesses=None,
        consequent_site="logits",
        consequent_value=obs_cache["logits"],
        consequent_scale=0.1,
        num_samples=50,
    )

    # extract_case_variables
    case_vars = extract_case_variables(traces)
    assert "head_0_0" in case_vars, "head_0_0 case variable missing"
    assert case_vars["head_0_0"].shape == torch.Size([50])
    print(f"Case values shape: {case_vars['head_0_0'].shape}")
    print(
        f"Case distribution: 0={(case_vars['head_0_0'] == 0).sum()}, 1={(case_vars['head_0_0'] == 1).sum()}, 2={(case_vars['head_0_0'] == 2).sum()}"
    )

    # compute_pns / necessity / sufficiency
    pns = compute_pns(case_vars["head_0_0"])
    nec = compute_necessity(case_vars["head_0_0"])
    suf = compute_sufficiency(case_vars["head_0_0"])
    assert 0.0 <= pns <= 1.0
    assert abs(pns - (nec + suf)) < 1e-6, "PNS should equal necessity + sufficiency"
    print(f"PNS={pns:.3f}, Necessity={nec:.3f}, Sufficiency={suf:.3f}")

    # summarize_verdict
    verdict = summarize_verdict(traces, antecedent_sites=["head_0_0", "head_0_1"])
    assert "head_0_0" in verdict
    assert "head_0_1" in verdict
    assert verdict["head_0_0"]["n_samples"] == 50
    assert verdict["head_0_1"]["n_samples"] == 0  # not an antecedent in this run
    print(f"head_0_0 verdict: {verdict['head_0_0']}")
    print(f"head_0_1 verdict (absent): {verdict['head_0_1']}")
    print("PASS: verdict.py correctly aggregates case variables into PNS estimates")
