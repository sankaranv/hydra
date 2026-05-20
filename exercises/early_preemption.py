"""
Early preemption: Alice/Bob deer-shooting (Fang 2022, §4.5.5).

Alice shoots at t=1:00 and the deer dies. Bob intended to shoot at t=1:01, but
because the deer is already dead he never fires. This contrasts with *late*
preemption (Sally/Billy): there, the backup's causal chain extends all the way
to the effect variable and is only cut off at the very end. Here, the backup
chain is cut off upstream — Bob never fires, because Alice's earlier shot
removes the precondition for Bob to act.

Consequence for witness choice: in late preemption the witness is a mediator
downstream of the backup throw (bill_hits). Here the witness is the backup
action itself (bob_shoots), which never fires. This reflects where in the
causal graph the preemption occurs.

Observations are conditioned directly on the actual world via condition(data=...)
rather than through Beta latent parameters, because the probabilities are baked
into the structural equations.
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
    alice_shoots = pyro.sample("alice_shoots", dist.Bernoulli(0.9))

    # Deer is dead by t=1:00 iff Alice shoots (hit probability = 1 when she shoots)
    deer_dead_100 = pyro.sample("deer_dead_100", dist.Bernoulli(alice_shoots))

    # Bob shoots iff the deer is not already dead and he has intention (p=0.9)
    bob_shoots = pyro.sample(
        "bob_shoots", dist.Bernoulli(0.9 * (1 - deer_dead_100))
    )

    # Deer is dead by t=1:01 if already dead, or if Bob shoots and hits (p=0.8)
    p_dead_101 = torch.clamp(
        deer_dead_100 + bob_shoots * 0.8 - deer_dead_100 * bob_shoots * 0.8,
        0.0,
        1.0,
    )
    deer_dead_101 = pyro.sample("deer_dead_101", dist.Bernoulli(p_dead_101))

    return {
        "alice_shoots": alice_shoots,
        "deer_dead_100": deer_dead_100,
        "bob_shoots": bob_shoots,
        "deer_dead_101": deer_dead_101,
    }


# Actual world: Alice shoots, deer dies at t=1:00, Bob does not shoot.
actual_obs = {
    "alice_shoots": torch.tensor(1.0),
    "deer_dead_100": torch.tensor(1.0),
    "bob_shoots": torch.tensor(0.0),
    "deer_dead_101": torch.tensor(1.0),
}

with ExtractSupports() as s:
    condition(model, data=actual_obs)()

NUM_SAMPLES = 10000
CONSEQUENT_SCALE = 1e-7
ANTECEDENT_BIAS = 0.1


def compute_logp(
    posterior: dict,
    antecedent_name: str,
    consequent_name: str,
    consequent_scale: float,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (raw_logp, conditional_logp).

    Per-sample factor = soft_neq(world1, proposed) + soft_eq(world2, proposed),
    computed from sample world values. World layout from SplitSubsets:
      world 0 = factual, world 1 = necessity (alternative), world 2 = sufficiency.
    """
    ant_key = f"__cause____antecedent_{antecedent_name}"
    ant = posterior[ant_key]
    cons = posterior[consequent_name]
    proposed = 1.0

    w1 = cons[:, 1].reshape(num_samples).float()
    w2 = cons[:, 2].reshape(num_samples).float()

    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))

    neq = torch.where(w1 != proposed, log1mc, logcs)
    eq = torch.where(w2 == proposed, log1mc, logcs)
    factor = neq + eq

    raw_logp = torch.log(torch.exp(factor).mean())
    mask = ant == 0
    cond_logp = torch.log(torch.exp(factor[mask]).mean())
    return raw_logp, cond_logp


def ac_check(logp: torch.Tensor) -> bool:
    """True when exp(logp) is clearly above the consequent_scale noise floor."""
    return torch.exp(logp).item() > 0.01


# ── Query A: Alice is the actual cause of deer_dead_101=1 ────────────────────
# Witness bob_shoots held at its factual value of 0. This prevents Bob from
# filling in as the cause in the counterfactual world where Alice doesn't shoot.
# Without the witness, Bob would be free to fire (deer not yet dead at t=1:00)
# and the deer might still die, making Alice's intervention look irrelevant.
#
# Contrast with late preemption: there the witness is bill_hits (a downstream
# mediator). Here it is bob_shoots (the backup action itself), because early
# preemption cuts the backup chain before it begins.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"alice_shoots": torch.tensor(1.0)},
        alternatives={"alice_shoots": torch.tensor(0.0)},
        witnesses={"bob_shoots": None},
        consequents={"deer_dead_101": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a:
        with condition(data={**actual_obs, **evidence_a}):
            posterior_a = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_alice_shoots", "deer_dead_101"],
            )()

logp_a, cond_logp_a = compute_logp(
    posterior_a, "alice_shoots", "deer_dead_101", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_a = ac_check(logp_a)

# Gather factual and counterfactual deer_dead_101 from Query A to make the
# world split explicit: in the world where Alice doesn't shoot and bob_shoots
# is pinned at 0, deer_dead_101 becomes 0.
d101_w0 = posterior_a["deer_dead_101"][:, 0].reshape(NUM_SAMPLES)
d101_w1 = posterior_a["deer_dead_101"][:, 1].reshape(NUM_SAMPLES)
ant_a = posterior_a["__cause____antecedent_alice_shoots"]
mask_a = ant_a == 0

print("Query A — Alice is the actual cause of deer_dead_101=1")
print(f"  ac_check                         = {ac_a}")
print(f"  exp(logp)                         = {torch.exp(logp_a).item():.4f}")
print(f"  deer_dead_101 factual (world 0)  = {d101_w0[mask_a].mean().item():.3f}  (expect 1.0)")
print(f"  deer_dead_101 alt    (world 1)   = {d101_w1[mask_a].mean().item():.3f}  (expect ~0.0 when witness pinned; >0 averaged over all ant==0 samples)")

# ── Query B: Bob's non-shooting is not a cause of deer_dead_101=1 ────────────
# Witness deer_dead_100 held at its factual value of 1. Even in the
# counterfactual where bob_shoots=1, the deer was already dead at t=1:00,
# so deer_dead_101=1 regardless. Bob's non-shooting made no difference.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"bob_shoots": torch.tensor(0.0)},
        alternatives={"bob_shoots": torch.tensor(1.0)},
        witnesses={"deer_dead_100": None},
        consequents={"deer_dead_101": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b:
        with condition(data={**actual_obs, **evidence_b}):
            posterior_b = Predictive(
                model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                + ["__cause____antecedent_bob_shoots", "deer_dead_101"],
            )()

logp_b, _ = compute_logp(
    posterior_b, "bob_shoots", "deer_dead_101", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_b = ac_check(logp_b)

print(f"\nQuery B — Bob's non-shooting is not a cause of deer_dead_101=1")
print(f"  ac_check   = {ac_b}")
print(f"  exp(logp)  = {torch.exp(logp_b).item():.2e}  (expect ~{CONSEQUENT_SCALE:.0e}, noise floor)")

assert ac_a, "Query A: Alice should be an actual cause"
assert not ac_b, "Query B: Bob's non-shooting should not be an actual cause"
print("\nAll assertions passed.")
