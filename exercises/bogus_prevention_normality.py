"""
Bogus prevention with normality: structural dependence (Halpern 2016, Example 3.4.2).

Distinct from Example 3.4.1: here Bodyguard and Assassin are structurally
linked. Assassin acts *only if* Bodyguard does not: assassin = NOT bodyguard.
The motivation is that Assassin is doing it to make Bodyguard look good.

Bodyguard puts in antidote (bodyguard=1). Assassin does not poison (assassin=0,
because NOT bodyguard = NOT 1 = 0). Victim survives.

Is Bodyguard a cause?

The HP analysis: to show necessity, the witness world must have assassin=0 when
bodyguard is removed. But assassin = NOT bodyguard means that in the world
where bodyguard=0, the structural equation gives assassin=1, not 0. The only
way to get assassin=0 in that world is to intervene on assassin directly,
which contradicts the structural equation. HP calls this witness world
"abnormal" and excludes it from consideration. Without it, necessity cannot
be demonstrated — Bodyguard is correctly judged NOT a cause.

ChiRho models normality via witness_bias: higher bias increases the probability
that witnesses are preempted to their factual values (assassin=0). This
prevents exploration of the structurally-determined counterfactual (assassin=1
when bodyguard=0), effectively implementing the abnormal-world exclusion.

Positive witness_bias: P(witness preempted to factual) = 0.5 + witness_bias.
- witness_bias=0.5: P=1.0 → witnesses always at factual → normal worlds only
- witness_bias=-0.4: P=0.1 → witnesses mostly follow structural eq → includes
  the abnormal world where assassin=1 given bodyguard=0

Two queries:

Query A (witness_bias=0.5): Bodyguard is NOT a cause.
  Assassin is always preempted to factual=0. In alt world (bodyguard=0),
  assassin=0 → victim_survives=1 unchanged. Necessity fails → False.
  This is the CORRECT answer — the structurally-forced assassin=1 world
  is excluded by the high bias toward factual witness values.

Query B (witness_bias=-0.4): Bodyguard IS a cause (wrong answer).
  Assassin mostly follows structural eq in alt world. When bodyguard=0,
  assassin = NOT(0) = 1 → victim_survives=0. Necessity holds → True.
  This is the WRONG answer — the abnormal witness world is included.
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


def bogus_normality():
    bodyguard = pyro.sample("bodyguard", dist.Bernoulli(0.5))
    # assassin = NOT bodyguard: acts only when bodyguard does not act.
    assassin = pyro.sample("assassin", dist.Bernoulli(1 - bodyguard))
    p_survive = torch.clamp((1 - assassin) + bodyguard, 0.0, 1.0)
    victim_survives = pyro.sample("victim_survives", dist.Bernoulli(p_survive))
    return {
        "bodyguard": bodyguard,
        "assassin": assassin,
        "victim_survives": victim_survives,
    }


actual_obs = {
    "bodyguard": torch.tensor(1.0),
    "assassin": torch.tensor(0.0),
    "victim_survives": torch.tensor(1.0),
}

with ExtractSupports() as s:
    condition(bogus_normality, data=actual_obs)()

NUM_SAMPLES = 20000
CONSEQUENT_SCALE = 1e-8
ANTECEDENT_BIAS = 0.1


def compute_logp(
    posterior: dict,
    consequent_scale: float,
    num_samples: int,
) -> torch.Tensor:
    """logp from single-antecedent world layout [N, 3, ...]."""
    cons = posterior["victim_survives"]
    w1 = cons[:, 1].reshape(num_samples).float()
    w2 = cons[:, 2].reshape(num_samples).float()
    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))
    neq = torch.where(w1 != 1.0, log1mc, logcs)
    eq = torch.where(w2 == 1.0, log1mc, logcs)
    return torch.log(torch.exp(neq + eq).mean())


def ac_check(logp: torch.Tensor) -> bool:
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    return True


# ── Query A: witness_bias=0.5 — Bodyguard is NOT a cause (correct) ───────────
# Assassin is always preempted to its factual value (0). With assassin=0 in
# all worlds, removing bodyguard leaves victim_survives=1 unchanged.
# The abnormal witness world (assassin=0 despite bodyguard=0) has probability 1.0;
# its normality exclusion is captured by pinning assassin to factual everywhere.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"bodyguard": torch.tensor(1.0)},
        alternatives={"bodyguard": torch.tensor(0.0)},
        witnesses={"assassin": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
        witness_bias=0.5,  # always preempt witnesses to factual
    ) as evidence_a:
        with condition(data={**actual_obs, **evidence_a}):
            posterior_a = Predictive(
                bogus_normality,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + ["__cause____antecedent_bodyguard", "victim_survives"],
            )()

logp_a = compute_logp(posterior_a, CONSEQUENT_SCALE, NUM_SAMPLES)
ac_a = ac_check(logp_a)

# ── Query B: witness_bias=-0.4 — Bodyguard IS a cause (wrong) ────────────────
# Assassin mostly follows its structural equation in alt world. When
# bodyguard=0 (alt), the structural equation gives assassin=NOT(0)=1,
# so victim_survives=NOT(1) OR 0=0 — necessity holds. This is the wrong
# answer: the structural equation creates an abnormal counterfactual
# (assassin poisoning precisely because bodyguard did not act) that
# should not count as a valid witness world.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"bodyguard": torch.tensor(1.0)},
        alternatives={"bodyguard": torch.tensor(0.0)},
        witnesses={"assassin": None},
        consequents={"victim_survives": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
        witness_bias=-0.4,  # rarely preempt: structural eq mostly applies
    ) as evidence_b:
        with condition(data={**actual_obs, **evidence_b}):
            posterior_b = Predictive(
                bogus_normality,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + ["__cause____antecedent_bodyguard", "victim_survives"],
            )()

logp_b = compute_logp(posterior_b, CONSEQUENT_SCALE, NUM_SAMPLES)
ac_b = ac_check(logp_b)

print(f"[Normal bias]   Bodyguard is cause: {ac_a}  # CORRECT — abnormal witness excluded")
print(f"[No preemption] Bodyguard is cause: {ac_b}   # WRONG — structural eq creates abnormal counterfactual")

assert not ac_a, "Query A: bodyguard should not be a cause at witness_bias=0.5"
assert ac_b, "Query B: bodyguard appears as cause at witness_bias=-0.4 (wrong answer)"
print("\nAll assertions passed.")
