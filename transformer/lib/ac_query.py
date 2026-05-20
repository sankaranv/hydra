"""SearchForExplanation wrapper for transformer actual causality queries.

The consequent is a named logit site. The AC query asks: "Did antecedent site A
(taking its observed activation value) cause the consequent site C (taking its
observed logit value)?"

Key design constraints:
- alternatives must be pre-specified (bypasses the Cauchy proposal that
  draws activations like -26.0, far outside the transformer activation range)
- consequent_scale must be calibrated to the output range (~0.1 works for
  unit-scale activations; increase for unnormalized logits)
- first_available_dim=-8 accommodates event_dim=2 sites ([batch, seq, d_model])
"""

from typing import Optional

import pyro
import pyro.distributions.constraints as constraints
import torch
from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers.explanation import SearchForExplanation
from chirho.observational.handlers.condition import condition


def logit_diff_condition(
    correct_token_id: int,
    incorrect_token_id: int,
    threshold: float = 0.0,
) -> callable:
    """Returns a factor function over a logit tensor site.

    The factor scores each world by whether the logit difference
    (correct - incorrect) exceeds the threshold. Passed as
    factors={"logits": logit_diff_condition(...)} to SearchForExplanation
    to replace the default soft-equality consequent.

    In the necessity world (case=1) the antecedent is ablated; the factor
    rewards a change in logit_diff. In the sufficiency world (case=2) the
    antecedent is restored; the factor rewards logit_diff being preserved.
    SearchForExplanation handles the world-level logic — this factor simply
    measures the consequent value in whatever world it is evaluated in.

    The factor is a soft indicator: exp(scale * (logit_diff - threshold)).
    SearchForExplanation wraps this in consequent_scale normalization
    internally, so the raw output should be in roughly unit scale.
    """

    def factor_fn(logits: torch.Tensor) -> torch.Tensor:
        # logits: [..., vocab_size] — index last dim for token logits
        logit_diff = logits[..., correct_token_id] - logits[..., incorrect_token_id]
        # Sum over all tokens and batch — produces a scalar log-factor
        return (logit_diff - threshold).sum()

    return factor_fn


def run_ac_query(
    model_fn,
    antecedents: dict[str, torch.Tensor],
    alternatives: dict[str, torch.Tensor],
    witnesses: Optional[dict[str, None]],
    consequent_site: str,
    consequent_value: torch.Tensor,
    consequent_scale: float = 0.1,
    num_samples: int = 200,
    first_available_dim: int = -8,
) -> list[dict]:
    """Run SearchForExplanation and return a list of trace dicts.

    Each trace dict maps site_name to sampled value, including the case
    variables (__cause____antecedent_SITE) that encode whether SITE is a cause:
      0 = factual world (not intervened)
      1 = necessity world (antecedent replaced with alternative)
      2 = sufficiency world (antecedent restored to observed value)

    The supports for antecedent and consequent sites are inferred from the
    observed tensors: multi-dimensional tensors use IndependentConstraint(real, N).
    This matches what ExtractSupports produces for pyro.deterministic sites.

    alternatives must cover all antecedent sites — this bypasses the default
    Cauchy proposal and keeps alternative values in the transformer activation range.

    Returns raw traces — use verdict.py to aggregate case variables into PNS.
    """

    # Build supports from the observed tensors.
    # pyro.deterministic with event_dim=N registers IndependentConstraint(real, N).
    # We reconstruct this directly from tensor ndim to avoid running ExtractSupports
    # (which would require a separate forward pass and may not see all sites).
    def tensor_support(t: torch.Tensor) -> constraints.Constraint:
        ndim = t.dim()
        if ndim == 0:
            return constraints.real
        return constraints.independent(constraints.real, ndim)

    antecedent_supports = {
        site: tensor_support(value) for site, value in antecedents.items()
    }
    consequent_support = {consequent_site: tensor_support(consequent_value)}
    supports = {**antecedent_supports, **consequent_support}

    traces = []
    for _ in range(num_samples):
        with MultiWorldCounterfactual(first_available_dim=first_available_dim):
            with SearchForExplanation(
                supports=supports,
                antecedents=antecedents,
                consequents={consequent_site: consequent_value},
                witnesses=witnesses,
                alternatives=alternatives,
                consequent_scale=consequent_scale,
            ) as evidence:
                with condition(data=evidence):
                    trace = pyro.poutine.trace(model_fn).get_trace()

        # Extract the full site dict for this sample
        traces.append(
            {
                name: node["value"]
                for name, node in trace.nodes.items()
                if node["type"] == "sample"
            }
        )

    return traces


if __name__ == "__main__":
    import pyro
    import torch.nn as nn

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

    print("=== ac_query.py validation ===")

    # Get observed values for antecedent and consequent
    obs_cache = run_and_cache(toy_model_fn, sites=["head_0_0", "logits"])
    obs_h00 = obs_cache["head_0_0"]
    obs_logits = obs_cache["logits"]

    # Zero-ablation as the alternative (in-range, interpretable)
    alt_h00 = zero_ablation({"head_0_0": obs_h00})

    traces = run_ac_query(
        model_fn=toy_model_fn,
        antecedents={"head_0_0": obs_h00},
        alternatives=alt_h00,
        witnesses=None,
        consequent_site="logits",
        consequent_value=obs_logits,
        consequent_scale=0.1,
        num_samples=30,
    )

    print(f"Collected {len(traces)} traces")
    assert len(traces) == 30, f"Expected 30 traces, got {len(traces)}"

    # Verify case variables are present and in {0, 1, 2}
    case_key = "__cause____antecedent_head_0_0"
    case_vals = [tr[case_key].item() for tr in traces if case_key in tr]
    assert len(case_vals) == 30, "Case variable missing from some traces"
    assert all(v in {0, 1, 2} for v in case_vals), (
        f"Unexpected case values: {set(case_vals)}"
    )

    case_counts = {v: case_vals.count(v) for v in {0, 1, 2}}
    pns = sum(1 for v in case_vals if v != 0) / len(case_vals)
    print(f"Case distribution: {case_counts}")
    print(f"PNS estimate (case != 0): {pns:.3f}")
    print("PASS: run_ac_query returns traces with valid case variables")
