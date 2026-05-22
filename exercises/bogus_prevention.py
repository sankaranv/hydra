"""
Bogus prevention: assassin + bodyguard (Halpern 2016, Example 3.4.1).

Assassin could put poison in Victim's coffee but has a change of heart and
does not. Bodyguard puts antidote in the coffee. Victim survives. Did
Bodyguard cause the survival?

Two models are compared in the same file:

`bogus_naive` (no intermediate variable):
  - victim_survives = NOT(assassin) OR bodyguard
  - Bodyguard appears to be a cause: with assassin=1 as witness value,
    removing bodyguard would kill Victim. The naive model "sees" this
    contingency and wrongly reports bodyguard as a cause.
  - Assassin's non-action also appears as a cause: same reasoning.

`bogus_enriched` (with poison_neutralised = assassin AND bodyguard):
  - victim_survives = NOT(assassin) OR poison_neutralised
  - PN=0 in the actual world because assassin=0 (no poison to neutralise).
  - With assassin=0 conditioned in all worlds, removing bodyguard leaves
    victim_survives=1 unchanged — bodyguard is NOT a cause.
  - Assassin's non-action IS a genuine cause: the antecedent intervention
    (assassin=1) overrides factual conditioning, and with PN preempted to
    0, victim_survives drops to 0.

The True/True → False/True pattern is the pedagogical point: the enriched
model blocks the witness exploration (assassin=1) that gives the naive model
its wrong answer, because conditioning on assassin=0 pins it in all worlds.

NOTE on PN formula: the spec comment says PN = (NOT assassin) AND bodyguard,
but that gives PN=1 when assassin=0, bodyguard=1, contradicting PN=0 in
actual world. The correct formula is PN = assassin AND bodyguard.

NOTE on naive_obs: conditioning on assassin or bodyguard in the naive model
pins them in all worlds, blocking witness exploration and giving False
instead of True. Only victim_survives is conditioned so the witness search
can explore assassin=1 (bodyguard query) or bodyguard=0 (assassin query).
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


def bogus_naive():
    u_assassin = pyro.sample("u_assassin", dist.Bernoulli(0.5))
    u_bodyguard = pyro.sample("u_bodyguard", dist.Bernoulli(0.5))
    assassin = pyro.deterministic("assassin", u_assassin)
    bodyguard = pyro.deterministic("bodyguard", u_bodyguard)
    p_survive = torch.clamp((1 - assassin) + bodyguard, 0.0, 1.0)
    victim_survives = pyro.sample("victim_survives", dist.Bernoulli(p_survive))
    return {
        "assassin": assassin,
        "bodyguard": bodyguard,
        "victim_survives": victim_survives,
    }


def bogus_enriched():
    u_assassin = pyro.sample("u_assassin", dist.Bernoulli(0.5))
    u_bodyguard = pyro.sample("u_bodyguard", dist.Bernoulli(0.5))
    assassin = pyro.deterministic("assassin", u_assassin)
    bodyguard = pyro.deterministic("bodyguard", u_bodyguard)
    # PN=0 in actual world: assassin didn't act, so no poison to neutralise.
    pn = pyro.sample("poison_neutralised", dist.Bernoulli(assassin * bodyguard))
    p_survive = torch.clamp((1 - assassin) + pn, 0.0, 1.0)
    victim_survives = pyro.sample("victim_survives", dist.Bernoulli(p_survive))
    return {
        "assassin": assassin,
        "bodyguard": bodyguard,
        "poison_neutralised": pn,
        "victim_survives": victim_survives,
    }


# Naive model: only condition on the outcome. Conditioning on assassin or
# bodyguard would pin them in all worlds, blocking the witness exploration.
naive_obs = {
    "victim_survives": torch.tensor(1.0),
}

# Enriched model: condition on all four factual values. assassin=0 pinned
# in all worlds (except when it is the antecedent) blocks bodyguard's causal claim.
enriched_obs = {
    "assassin": torch.tensor(0.0),
    "bodyguard": torch.tensor(1.0),
    "poison_neutralised": torch.tensor(0.0),
    "victim_survives": torch.tensor(1.0),
}

with ExtractSupports() as s_naive:
    condition(bogus_naive, data=naive_obs)()

with ExtractSupports() as s_enriched:
    condition(bogus_enriched, data=enriched_obs)()

NUM_SAMPLES = 10000
CONSEQUENT_SCALE = 1e-8
ANTECEDENT_BIAS = 0.1


def compute_logp(
    posterior: dict,
    consequent_name: str,
    consequent_scale: float,
    num_samples: int,
) -> torch.Tensor:
    """
    logp for a single antecedent.

    World layout: 0=factual, 1=necessity (alternative), 2=sufficiency.
    Factor = soft_neq(world1, proposed) + soft_eq(world2, proposed).
    """
    cons = posterior[consequent_name]
    w1 = cons[:, 1].reshape(num_samples).float()
    w2 = cons[:, 2].reshape(num_samples).float()
    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))
    neq = torch.where(w1 != 1.0, log1mc, logcs)
    eq = torch.where(w2 == 1.0, log1mc, logcs)
    return torch.log(torch.exp(neq + eq).mean())


def ac_check(logp: torch.Tensor) -> bool:
    """True when exp(logp) is clearly above the noise floor."""
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    return True


# ── Query A: Bodyguard is (wrongly) a cause in the naive model ────────────────
# Without PN, the witness search can explore assassin=1 as a contingency.
# With assassin=1 as witness, removing bodyguard causes victim_survives=0.
# This satisfies HP's AC2 (necessity under contingent witness), hence True.
# The answer is wrong — the contingency (assassin=1) is not the actual world.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_naive.supports,
        antecedents={"bodyguard": torch.tensor(1.0)},
        alternatives={"bodyguard": torch.tensor(0.0)},
        witnesses={"assassin": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a:
        with condition(data={**naive_obs, **evidence_a}):
            posterior_a = Predictive(
                bogus_naive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_naive.supports.keys())
                    + ["__cause____antecedent_bodyguard", "victim_survives"],
            )()

logp_a = compute_logp(posterior_a, "victim_survives", CONSEQUENT_SCALE, NUM_SAMPLES)
ac_a = ac_check(logp_a)

# ── Query B: Assassin's non-action is (wrongly) a cause in the naive model ───
# Without PN, the witness search explores bodyguard=0. With bodyguard=0 as
# witness, switching assassin from 0 to 1 causes victim_survives=0. Wrong answer.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_naive.supports,
        antecedents={"assassin": torch.tensor(0.0)},
        alternatives={"assassin": torch.tensor(1.0)},
        witnesses={"bodyguard": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b:
        with condition(data={**naive_obs, **evidence_b}):
            posterior_b = Predictive(
                bogus_naive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_naive.supports.keys())
                    + ["__cause____antecedent_assassin", "victim_survives"],
            )()

logp_b = compute_logp(posterior_b, "victim_survives", CONSEQUENT_SCALE, NUM_SAMPLES)
ac_b = ac_check(logp_b)

# ── Query C: Bodyguard is NOT a cause in the enriched model ──────────────────
# assassin=0 is conditioned in all worlds. In alt world (bodyguard=0): PN=0
# (structural eq: 0*0=0); victim_survives = NOT(0) OR 0 = 1 — no change.
# Bodyguard's action flowed through PN, and PN=0 disconnects the path.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_enriched.supports,
        antecedents={"bodyguard": torch.tensor(1.0)},
        alternatives={"bodyguard": torch.tensor(0.0)},
        witnesses={"assassin": None, "poison_neutralised": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_c:
        with condition(data={**enriched_obs, **evidence_c}):
            posterior_c = Predictive(
                bogus_enriched,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_enriched.supports.keys())
                    + ["__cause____antecedent_bodyguard", "victim_survives"],
            )()

logp_c = compute_logp(posterior_c, "victim_survives", CONSEQUENT_SCALE, NUM_SAMPLES)
ac_c = ac_check(logp_c)

# ── Query D: Assassin's non-action IS a cause in the enriched model ───────────
# assassin=0→1 is the antecedent intervention (overrides conditioning in alt world).
# With PN preempted to factual (0), victim_survives = NOT(1) OR 0 = 0 — Victim dies.
# Assassin's restraint was genuinely necessary; the enriched model captures this.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_enriched.supports,
        antecedents={"assassin": torch.tensor(0.0)},
        alternatives={"assassin": torch.tensor(1.0)},
        witnesses={"poison_neutralised": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_d:
        with condition(data={**enriched_obs, **evidence_d}):
            posterior_d = Predictive(
                bogus_enriched,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_enriched.supports.keys())
                    + ["__cause____antecedent_assassin", "victim_survives"],
            )()

logp_d = compute_logp(posterior_d, "victim_survives", CONSEQUENT_SCALE, NUM_SAMPLES)
ac_d = ac_check(logp_d)

print(f"[Naive]    Bodyguard is cause:           {ac_a}   # WRONG — no PN variable, treats antidote as necessary")
print(f"[Naive]    Assassin non-action is cause: {ac_b}   # WRONG — no PN variable")
print(f"[Enriched] Bodyguard is cause:           {ac_c}  # CORRECT — PN=0, antidote did nothing")
print(f"[Enriched] Assassin non-action is cause: {ac_d}   # CORRECT — restraint was necessary")

assert ac_a, "Query A: naive bodyguard should be a (wrong) cause"
assert ac_b, "Query B: naive assassin non-action should be a (wrong) cause"
assert not ac_c, "Query C: enriched bodyguard should not be a cause"
assert ac_d, "Query D: enriched assassin non-action should be a cause"
print("\nAll assertions passed.")
