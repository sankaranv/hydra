"""
Late preemption: Sally/Billy rock-throwing (Halpern 2016, §2.3.3).

Sally's rock arrives first and shatters the bottle. Billy's rock would have
shattered it had Sally not thrown, but it arrives after the bottle is already
gone. Naive but-for analysis fails for Sally: if Sally hadn't thrown, Billy
would have hit — so the bottle would still have shattered. The fix is the
witness `bill_hits`: hold it at its factual value of 0, and now Sally's
counterfactual intervention makes a difference. Billy is correctly diagnosed
as not a cause because his hit is 0 in the actual world.

The ChiRho reference implementation wraps throw-to-hit probabilities in
Beta(1,1) latents masked from the likelihood. Conditioning all six prob_*
to 1.0 recovers the deterministic story.
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
    prob_sally_throws = pyro.sample("prob_sally_throws", dist.Beta(1, 1))
    prob_bill_throws = pyro.sample("prob_bill_throws", dist.Beta(1, 1))
    prob_sally_hits = pyro.sample("prob_sally_hits", dist.Beta(1, 1))
    prob_bill_hits = pyro.sample("prob_bill_hits", dist.Beta(1, 1))
    prob_shatters_if_sally = pyro.sample("prob_shatters_if_sally", dist.Beta(1, 1))
    prob_shatters_if_bill = pyro.sample("prob_shatters_if_bill", dist.Beta(1, 1))

    # Structural equations (probabilistic, deterministic when all prob_* = 1)
    sally_throws = pyro.sample("sally_throws", dist.Bernoulli(prob_sally_throws))
    bill_throws = pyro.sample("bill_throws", dist.Bernoulli(prob_bill_throws))
    sally_hits = pyro.sample(
        "sally_hits", dist.Bernoulli(prob_sally_hits * sally_throws)
    )
    # Preemption clause: Billy can only hit if Sally didn't
    bill_hits = pyro.sample(
        "bill_hits",
        dist.Bernoulli(prob_bill_hits * bill_throws * (1 - sally_hits)),
    )
    p_shatters = torch.clamp(
        prob_shatters_if_sally * sally_hits
        + prob_shatters_if_bill * bill_hits
        - prob_shatters_if_sally * prob_shatters_if_bill * sally_hits * bill_hits,
        0.0,
        1.0,
    )
    bottle_shatters = pyro.sample("bottle_shatters", dist.Bernoulli(p_shatters))
    return {
        "sally_throws": sally_throws,
        "bill_throws": bill_throws,
        "sally_hits": sally_hits,
        "bill_hits": bill_hits,
        "bottle_shatters": bottle_shatters,
    }


# Conditioning all prob_* to 1.0 gives the deterministic all-ones story:
# Sally throws → hits → shatters; Billy throws but does not hit.
deterministic_obs = {
    k: torch.tensor(1.0)
    for k in [
        "prob_sally_throws",
        "prob_bill_throws",
        "prob_sally_hits",
        "prob_bill_hits",
        "prob_shatters_if_sally",
        "prob_shatters_if_bill",
    ]
}

with ExtractSupports() as s:
    condition(model, data=deterministic_obs)()

NUM_SAMPLES = 10000
CONSEQUENT_SCALE = 1e-5


def compute_logp(
    posterior: dict,
    antecedent_name: str,
    consequent_name: str,
    consequent_scale: float,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (raw_logp, conditional_logp).

    raw_logp: log mean_exp of per-sample factor across all samples.
    conditional_logp: same, restricted to samples where the antecedent
      intervention was applied (antecedent preemption variable == 0).

    The per-sample factor is soft_neq(world1, proposed) + soft_eq(world2,
    proposed), computed from sample world values because the ChiRho factor
    site carries a zero-sized indexed dimension under Predictive and cannot
    be summed directly. The math is identical to what the working factor site
    would produce.

    World layout from SplitSubsets + MultiWorldCounterfactual:
      world 0 = factual, world 1 = necessity (alternative), world 2 = sufficiency.
    """
    ant_key = f"__cause____antecedent_{antecedent_name}"
    ant = posterior[ant_key]  # [N], values in {0, 1}
    cons = posterior[consequent_name]  # [N, 3, ...]
    proposed = 1.0

    w1 = cons[:, 1].reshape(num_samples).float()  # necessity world
    w2 = cons[:, 2].reshape(num_samples).float()  # sufficiency world

    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))

    # soft_neq(bool, w1, proposed): near-0 if w1 differs, large-negative if equal
    neq = torch.where(w1 != proposed, log1mc, logcs)
    # soft_eq(bool, w2, proposed): near-0 if w2 matches, large-negative if differs
    eq = torch.where(w2 == proposed, log1mc, logcs)

    factor = neq + eq  # [N]

    raw_logp = torch.log(torch.exp(factor).mean())
    mask = ant == 0  # intervention was applied (preemption did not fire)
    cond_logp = torch.log(torch.exp(factor[mask]).mean())
    return raw_logp, cond_logp


def ac_check(logp: torch.Tensor) -> bool:
    """True when exp(logp) is clearly above the consequent_scale noise floor."""
    return torch.exp(logp).item() > 0.01


# ── Query A: Sally is an actual cause ────────────────────────────────────────
# Witness bill_hits held at its factual value of 0. This prevents the backup
# causal chain from activating in the counterfactual world where Sally doesn't
# throw: without the witness, Billy would hit and the bottle would still shatter,
# making Sally's intervention look irrelevant.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"sally_throws": torch.tensor(1.0)},
        alternatives={"sally_throws": torch.tensor(0.0)},
        witnesses={"bill_hits": None},
        consequents={"bottle_shatters": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
    ) as evidence_a:
        with condition(data={**deterministic_obs, **evidence_a}):
            posterior_a = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_sally_throws", "bottle_shatters"],
            )()

logp_a, cond_logp_a = compute_logp(
    posterior_a, "sally_throws", "bottle_shatters", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_a = ac_check(logp_a)

print(f"Query A — Sally is an actual cause")
print(f"  ac_check        = {ac_a}")
print(f"  exp(logp)       = {torch.exp(logp_a).item():.4f}  (expect ~0.25)")
print(f"  exp(cond_logp)  = {torch.exp(cond_logp_a).item():.4f}  (expect ~0.50, conditioned on intervention applied)")

# ── Query B: Billy is not an actual cause ────────────────────────────────────
# Witness sally_hits held at its factual value of 1. Even if Billy's throw is
# removed in the counterfactual, the bottle still shatters because Sally hit it.
# So the consequent does not change — Billy's throw is not a but-for cause.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"bill_throws": torch.tensor(1.0)},
        alternatives={"bill_throws": torch.tensor(0.0)},
        witnesses={"sally_hits": None},
        consequents={"bottle_shatters": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
    ) as evidence_b:
        with condition(data={**deterministic_obs, **evidence_b}):
            posterior_b = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_bill_throws", "bottle_shatters"],
            )()

logp_b, cond_logp_b = compute_logp(
    posterior_b, "bill_throws", "bottle_shatters", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_b = ac_check(logp_b)

print(f"\nQuery B — Billy is not an actual cause")
print(f"  ac_check        = {ac_b}")
print(f"  exp(logp)       = {torch.exp(logp_b).item():.2e}  (expect ~{CONSEQUENT_SCALE:.0e}, noise floor)")

assert ac_a, "Query A: Sally should be an actual cause"
assert not ac_b, "Query B: Billy should not be an actual cause"
print("\nAll assertions passed.")
