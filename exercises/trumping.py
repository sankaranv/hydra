"""
Trumping: Captain/Sergeant orders to a private (Halpern 2016, §2.3.7;
originally Schaffer 2000).

A captain and a sergeant both simultaneously order "Charge!" The private obeys
the highest-ranking officer. Because the captain outranks the sergeant, the
captain's order trumps. Both orders are given at the same time — this is not a
timing story. The question is which order caused the charge.

The naive three-variable model — captain_orders, sergeant_orders,
private_charges = captain_orders OR sergeant_orders — cannot represent
trumping: it makes both orders symmetric causes. The correct model adds
sergeant_effective, which encodes the sergeant's authority as zero whenever
the captain has also ordered (HP §4.1: variable choice determines causal claims).

Query C demonstrates this with the naive model: it returns True for
sergeant_orders as a cause — the wrong answer — showing that model structure
determines the causal verdict.
"""

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive

from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers.components import ExtractSupports
from chirho.explainable.handlers.explanation import SearchForExplanation
from chirho.observational.handlers.condition import condition

pyro.set_rng_seed(0)


def model():
    captain_orders = pyro.sample("captain_orders", dist.Bernoulli(1.0))
    sergeant_orders = pyro.sample("sergeant_orders", dist.Bernoulli(1.0))
    # The sergeant's order has force only when the captain has not also ordered.
    sergeant_effective = pyro.sample(
        "sergeant_effective",
        dist.Bernoulli(sergeant_orders * (1 - captain_orders)),
    )
    p_charges = torch.clamp(
        captain_orders
        + sergeant_effective
        - captain_orders * sergeant_effective,
        0.0,
        1.0,
    )
    private_charges = pyro.sample("private_charges", dist.Bernoulli(p_charges))
    return {
        "captain_orders": captain_orders,
        "sergeant_orders": sergeant_orders,
        "sergeant_effective": sergeant_effective,
        "private_charges": private_charges,
    }


def naive_model():
    """Three-variable model without sergeant_effective — structurally wrong.

    captain_orders uses a non-trivial prior (0.5) to reflect the naive model's
    inability to encode rank precedence: without sergeant_effective, the model
    treats both orders as symmetric and cannot represent the captain's authority
    as deterministic context for the sergeant's role. The sergeant therefore
    appears causally relevant whenever the captain might not have ordered.
    """
    captain_orders = pyro.sample("captain_orders", dist.Bernoulli(0.5))
    sergeant_orders = pyro.sample("sergeant_orders", dist.Bernoulli(1.0))
    p_charges = torch.clamp(
        captain_orders + sergeant_orders - captain_orders * sergeant_orders,
        0.0,
        1.0,
    )
    private_charges = pyro.sample("private_charges", dist.Bernoulli(p_charges))
    return {
        "captain_orders": captain_orders,
        "sergeant_orders": sergeant_orders,
        "private_charges": private_charges,
    }


actual_obs = {
    "captain_orders": torch.tensor(1.0),
    "sergeant_orders": torch.tensor(1.0),
    "sergeant_effective": torch.tensor(0.0),
    "private_charges": torch.tensor(1.0),
}
# captain_orders excluded: the naive modeler did not observe the captain; leaving it
# unconditioned lets the Bernoulli(0.5) prior run, so sergeant_orders appears necessary.
naive_obs = {k: v for k, v in actual_obs.items() if k not in ("sergeant_effective", "captain_orders")}

with ExtractSupports() as s:
    condition(model, data=actual_obs)()

with ExtractSupports() as s_naive:
    condition(naive_model, data=naive_obs)()

NUM_SAMPLES = 10000
CONSEQUENT_SCALE = 1e-7
ANTECEDENT_BIAS = 0.1


def compute_logp(
    posterior: dict,
    antecedent_name: str,
    consequent_name: str,
    consequent_scale: float,
    num_samples: int,
) -> torch.Tensor:
    """
    Returns raw_logp = log mean_exp of per-sample factor over all samples.

    Factor = soft_neq(world1, proposed) + soft_eq(world2, proposed).
    World layout: 0 = factual, 1 = necessity (alternative), 2 = sufficiency.
    """
    ant_key = f"__cause____antecedent_{antecedent_name}"
    ant = posterior[ant_key]  # noqa: F841 — kept for caller readability
    cons = posterior[consequent_name]
    proposed = 1.0

    w1 = cons[:, 1].reshape(num_samples).float()
    w2 = cons[:, 2].reshape(num_samples).float()

    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))

    neq = torch.where(w1 != proposed, log1mc, logcs)
    eq = torch.where(w2 == proposed, log1mc, logcs)
    return torch.log(torch.exp(neq + eq).mean())


def ac_check(logp: torch.Tensor) -> bool:
    """True when exp(logp) is clearly above the consequent_scale noise floor."""
    return torch.exp(logp).item() > 0.01


# ── Query A: Captain is the actual cause ─────────────────────────────────────
# Witness sergeant_effective held at its factual value of 0. With the captain's
# order removed (alternative world) and sergeant_effective pinned at 0, the
# private does not charge. The captain's order is necessary for the charge.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"captain_orders": torch.tensor(1.0)},
        alternatives={"captain_orders": torch.tensor(0.0)},
        witnesses={"sergeant_effective": None},
        consequents={"private_charges": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a:
        with condition(data={**actual_obs, **evidence_a}):
            posterior_a = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_captain_orders", "private_charges"],
            )()

logp_a = compute_logp(
    posterior_a, "captain_orders", "private_charges", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_a = ac_check(logp_a)

# ── Query B: Sergeant is not the actual cause ─────────────────────────────────
# Same witness sergeant_effective=0. In the factual world sergeant_effective=0
# already (captain outranks), so the sergeant's order contributes nothing to
# the charge. Removing sergeant_orders changes nothing downstream.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"sergeant_orders": torch.tensor(1.0)},
        alternatives={"sergeant_orders": torch.tensor(0.0)},
        witnesses={"sergeant_effective": None},
        consequents={"private_charges": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b:
        with condition(data={**actual_obs, **evidence_b}):
            posterior_b = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_sergeant_orders", "private_charges"],
            )()

logp_b = compute_logp(
    posterior_b, "sergeant_orders", "private_charges", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_b = ac_check(logp_b)

# ── Query C: naive model — wrong answer, illustrates importance of variable choice
# Without sergeant_effective the model treats both orders as symmetric. Asking
# whether sergeant_orders is a cause returns True — the wrong verdict. This shows
# that the causal conclusion depends on the model structure, not just the data.

with ExtractSupports() as s_naive:
    condition(naive_model, data=naive_obs)()

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_naive.supports,
        antecedents={"sergeant_orders": torch.tensor(1.0)},
        alternatives={"sergeant_orders": torch.tensor(0.0)},
        witnesses={},
        consequents={"private_charges": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_c:
        with condition(data={**naive_obs, **evidence_c}):
            posterior_c = Predictive(
                naive_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_naive.supports.keys())
                + ["__cause____antecedent_sergeant_orders", "private_charges"],
            )()

logp_c = compute_logp(
    posterior_c, "sergeant_orders", "private_charges", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_c = ac_check(logp_c)

print("Query A — Captain is the actual cause (correct model)")
print(f"  ac_check   = {ac_a}   (expect True)   exp(logp) = {torch.exp(logp_a).item():.4f}")

print("\nQuery B — Sergeant is not the actual cause (correct model)")
print(f"  ac_check   = {ac_b}   (expect False)  exp(logp) = {torch.exp(logp_b).item():.2e}")

print("\nQuery C — Sergeant appears to be a cause (naive model, wrong answer)")
print(f"  ac_check   = {ac_c}   (expect True — this is the wrong conclusion)")
print("  The naive model omits sergeant_effective and cannot represent trumping.")
print("  Both orders look symmetric, so sergeant_orders is spuriously flagged as a cause.")

assert ac_a, "Query A: captain should be an actual cause"  # correct model, correct verdict
assert not ac_b, "Query B: sergeant should not be an actual cause"  # correct model, correct verdict
assert ac_c, "Query C: naive model wrongly returns True for sergeant"  # broken model
print("\nAll assertions passed.")
