"""SearchForExplanation wrapper for transformer actual causality queries.

The consequent is a named logit site. The AC query asks: "Did antecedent site A
(taking its observed activation value) cause the consequent site C (taking its
observed logit value)?"

Key design constraints:
- alternatives must be pre-specified (bypasses the Cauchy proposal that
  draws activations like -26.0, far outside the transformer activation range)
- consequent_scale must be calibrated to the output range (~0.1 works for
  unit-scale activations; increase for unnormalized logits)
- event_dim=2 throughout: all HookedGPT2 sites are [batch, seq, d_model];
  supports are built as IndependentConstraint(real, 2) to match the registration
- first_available_dim=-8 accommodates event_dim=2 sites plus world dims

Case variable conventions — two independent code paths use different semantics:

  run_ac_query (Monte Carlo path): samples from Categorical(3) per the chirho
  tutorial convention. case=0 = factual world, case=1 = necessity world,
  case=2 = sufficiency world. PNS = P(case != 0). Used by summarize_verdict().

  train_ac_guide (IS path): samples from the Categorical(2) binary prior
  introduced by chirho 0.3.0's Preemptions handler. case=0 = intervention NOT
  preempted (MultiWorldCounterfactual fires; necessity + sufficiency worlds
  active), case=1 = intervention preempted (undo_split; factual world only).
  PNS = P(case=0 | evidence). Used by read_guide_verdict().

Guide design (train_ac_guide):
  SVI+REINFORCE is numerically unstable when soft_neq(v=v) = log(0) = -inf,
  which occurs for non-causal sites where ablation leaves the consequent unchanged.
  Instead, train_ac_guide uses importance sampling: n_steps forward passes from
  the Categorical(2) prior, re-weighted by softmax(log_prob_sums). Samples with
  -inf log_prob get weight 0 via softmax (no NaN). IS-weighted P(case=0 | evidence)
  per site is returned directly — no param_store side effects.
"""

from typing import Callable, Optional

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
) -> Callable[[torch.Tensor], torch.Tensor]:
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
    model_fn: Callable,
    *model_args,
    antecedents: dict[str, torch.Tensor],
    alternatives: dict[str, torch.Tensor],
    witnesses: Optional[dict[str, None]],
    consequent_site: str,
    consequent_value: torch.Tensor,
    event_dim: int = 2,
    consequent_scale: float = 0.1,
    num_samples: int = 200,
    first_available_dim: int = -8,
) -> list[dict]:
    """Run SearchForExplanation and return a list of trace dicts.

    model_fn is called as model_fn(*model_args) on each sample. For models
    that close over their inputs (common in AC queries where the observed
    activation values are baked in), pass no model_args. For models that
    accept input tensors, pass them as positional arguments after model_fn.

    Each trace dict maps site_name to sampled value, including the case
    variables (__cause____antecedent_SITE) that encode whether SITE is a cause:
      0 = factual world (not intervened)
      1 = necessity world (antecedent replaced with alternative)
      2 = sufficiency world (antecedent restored to observed value)

    event_dim is the registration event_dim used for all pyro.deterministic sites.
    For HookedGPT2 sites ([batch, seq, d_model] tensors), event_dim=2 throughout.
    This must match the event_dim used when registering the sites — a mismatch
    causes gather/undo_split inside SearchForExplanation to treat batch dimensions
    as event dimensions, silently corrupting world indices.

    alternatives must cover all antecedent sites — this bypasses the default
    Cauchy proposal and keeps alternative values in the transformer activation range.

    Each of the num_samples calls is a separate independent forward pass;
    SearchForExplanation's per-sample case variable draws are not vectorized
    because each sample needs an independent Pyro trace.

    Returns raw traces — use verdict.py to aggregate case variables into PNS.
    """

    # Build supports to match how pyro.deterministic registered the sites.
    # HookedGPT2 uses event_dim=2 for all sites; the support must be
    # IndependentConstraint(real, 2) — not t.dim() which would include batch dims.
    def site_support(t: torch.Tensor) -> constraints.Constraint:
        if event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, event_dim)

    antecedent_supports = {site: site_support(v) for site, v in antecedents.items()}
    consequent_support = {consequent_site: site_support(consequent_value)}
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
                    trace = pyro.poutine.trace(model_fn).get_trace(*model_args)

        traces.append(
            {
                name: node["value"]
                for name, node in trace.nodes.items()
                if node["type"] == "sample"
            }
        )

    return traces


def train_ac_guide(
    model_fn: Callable,
    *model_args,
    antecedents: dict[str, torch.Tensor],
    alternatives: dict[str, torch.Tensor],
    witnesses: Optional[dict[str, None]],
    consequent_site: str,
    consequent_value: torch.Tensor,
    event_dim: int = 2,
    consequent_scale: float = 0.1,
    first_available_dim: int = -8,
    n_steps: int = 300,
    prefix: str = "__cause__",
) -> tuple[list[float], dict[str, float]]:
    """Estimate per-site PNS via importance sampling (IS) from the prior.

    SVI+REINFORCE is numerically unstable when soft_neq(v=v) = log(0) = -inf,
    which occurs for non-causal sites where ablation leaves the consequent
    unchanged. IS handles -inf via softmax: those samples get weight 0 and
    contribute nothing to the IS-weighted posterior.

    Runs n_steps forward passes inside the SearchForExplanation context. Each
    pass samples case variables from the Categorical(2) prior (case=0: intervention
    active, case=1: factual/preempted). The trace log_prob_sum captures how well
    each sample explains the observed consequent. IS weights = softmax(log_prob_sums).

    IS-weighted P(case=0 | evidence) is computed per antecedent site. When ALL
    log_probs are -inf (necessity condition never fires — the antecedent does not
    change the consequent in any world), softmax would give nan (0/0); the function
    detects this case and defaults to P(case=0) = 0 (non-causal).

    For a causal site: case=0 samples satisfy the necessity factor (soft_neq is
    high because ablation changes the consequent) → high IS weight → P(case=0) > 0.5.
    For a non-causal site: case=0 samples have necessity factor = -inf → all
    log_probs are -inf → all-inf fallback → P(case=0) ≈ 0.

    Returns (log_probs, pns_dict) where pns_dict maps site_name →
    P(case=0 | evidence). Pass pns_dict to read_guide_verdict() for the
    formatted verdict. No global state is modified.
    """

    def site_support(t: torch.Tensor) -> constraints.Constraint:
        if event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, event_dim)

    supports = {
        **{site: site_support(v) for site, v in antecedents.items()},
        consequent_site: site_support(consequent_value),
    }

    # Construct the case variable name that SearchForExplanation uses per antecedent.
    case_var_name = {site: f"{prefix}__antecedent_{site}" for site in antecedents}

    # Run n_steps prior samples, collecting case value and log_prob_sum per sample.
    log_probs: list[float] = []
    # site -> list of (sample_index, case_value) for samples that contained the case var
    site_case_records: dict[str, list[tuple[int, int]]] = {s: [] for s in antecedents}

    for i in range(n_steps):
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
                    trace = pyro.poutine.trace(model_fn).get_trace(*model_args)

        log_probs.append(trace.log_prob_sum().item())

        for site in antecedents:
            cv = case_var_name[site]
            if cv in trace.nodes:
                site_case_records[site].append(
                    (i, int(trace.nodes[cv]["value"].item()))
                )

    # IS weights via softmax — exp(-inf) = 0, so -inf samples are excluded automatically.
    # When ALL log_probs are -inf, necessity never fired: the antecedent did not change
    # the consequent in any world. Softmax would give nan (0/0); default to zero weights.
    log_probs_t = torch.tensor(log_probs, dtype=torch.float64)
    all_inf = not log_probs_t.isfinite().any().item()
    is_weights = (
        torch.zeros(len(log_probs), dtype=torch.float32)
        if all_inf
        else torch.softmax(log_probs_t, dim=0).float()
    )

    # IS-weighted P(case=0 | evidence) per antecedent site.
    pns_dict: dict[str, float] = {}
    for site in antecedents:
        records = site_case_records[site]
        if not records:
            pns_dict[site] = float("nan")
            continue

        p0 = float(sum(is_weights[idx].item() for idx, case in records if case == 0))
        # Clamp to (0, 1) to keep the value meaningful as a probability.
        p0 = max(p0, 1e-6)
        p0 = min(p0, 1.0 - 1e-6)
        pns_dict[site] = p0

    return log_probs, pns_dict
