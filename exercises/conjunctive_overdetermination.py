"""
Conjunctive overdetermination: lightning and match-dropping (conjunctive case).

This file isolates the conjunctive model (forest_fire = match AND lightning) for
deeper study. The conjunctive model has two pedagogical layers:

1. Both singletons are actual causes (Queries A and B): unlike the disjunctive
   model, but-for holds for each singleton directly. Neither preempts the other;
   both are symmetric necessary contributors.

2. The joint set {match, lightning} fails AC3 (Query C): because each singleton
   already satisfies AC2, the joint set is overcomplete. This is HP's minimality
   condition (AC3) doing its job — it rules out cause sets that contain redundant
   members.

3. Context sensitivity (Query D): in a different context where lightning is absent
   (lightning=0), forest_fire=0. Asking whether match caused forest_fire=1
   violates AC1: the consequent did not occur. The sufficiency check fails because
   with lightning=0, holding match at its factual value does not produce fire.
   This is a different kind of failure from Query C — AC3 vs AC1.
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


def ff_conjunctive():
    u_match_dropped = pyro.sample("u_match_dropped", dist.Bernoulli(0.5))
    u_lightning = pyro.sample("u_lightning", dist.Bernoulli(0.5))
    match_dropped = pyro.deterministic("match_dropped", u_match_dropped)
    lightning = pyro.deterministic("lightning", u_lightning)
    p_fire = match_dropped * lightning
    forest_fire = pyro.sample("forest_fire", dist.Bernoulli(p_fire))
    return {
        "match_dropped": match_dropped,
        "lightning": lightning,
        "forest_fire": forest_fire,
    }


# Both causes present and fire burns.
obs_both = {
    "match_dropped": torch.tensor(1.0),
    "lightning": torch.tensor(1.0),
    "forest_fire": torch.tensor(1.0),
}

# Context for Query D: lightning absent, fire did not happen.
obs_lightning_absent = {
    "match_dropped": torch.tensor(1.0),
    "lightning": torch.tensor(0.0),
    "forest_fire": torch.tensor(0.0),
}

with ExtractSupports() as s:
    condition(ff_conjunctive, data=obs_both)()

NUM_SAMPLES = 10000
CONSEQUENT_SCALE = 1e-8
ANTECEDENT_BIAS = 0.1


def compute_logp_single(
    posterior: dict,
    antecedent_name: str,
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


def compute_logp_joint(
    posterior: dict,
    antecedent_names: tuple,
    consequent_name: str,
    consequent_scale: float,
    num_samples: int,
) -> tuple:
    """
    logp for a 2-element joint antecedent set, with AC3 minimality check.

    World dims: dim 1 = first antecedent, dim 2 = second antecedent.
    When ant_b is preempted (case=1), dim 2 collapses to b's factual value,
    making the factor reflect the singleton-a contribution. This allows the
    AC3 check without separate queries.

    Returns (logp, not_minimal).
    """
    a_name, b_name = antecedent_names
    ant_a = posterior[f"__cause____antecedent_{a_name}"]
    ant_b = posterior[f"__cause____antecedent_{b_name}"]
    cons = posterior[consequent_name]
    w1 = cons[:, 1, 1].reshape(num_samples).float()
    w2 = cons[:, 2, 2].reshape(num_samples).float()
    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))
    neq = torch.where(w1 != 1.0, log1mc, logcs)
    eq = torch.where(w2 == 1.0, log1mc, logcs)
    factor = neq + eq
    logp = torch.log(torch.exp(factor).mean())

    def _singleton_p(mask: torch.Tensor) -> float:
        if mask.sum() == 0:
            return 0.0
        return torch.exp(torch.log(torch.exp(factor[mask]).mean())).item()

    mask_a_only = (ant_a == 0) & (ant_b == 1)
    mask_b_only = (ant_a == 1) & (ant_b == 0)
    not_minimal = _singleton_p(mask_a_only) > 0.01 or _singleton_p(mask_b_only) > 0.01
    return logp, not_minimal


def ac_check(logp: torch.Tensor, not_minimal: bool = False) -> bool:
    """
    True when necessity and sufficiency hold and the antecedent set is minimal.

    Two distinct failure modes print different messages:
    - Noise floor: necessity or sufficiency failed outright (no causal signal).
    - not_minimal: AC3 violated (a proper subset is already a cause).
    """
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    if not_minimal:
        print("  The antecedent set is not minimal.")
        return False
    return True


# ── Query A: match_dropped is a cause (both=1 context) ───────────────────────
# Lightning held at factual (1). Removing match drops match*lightning from 1 to
# 0 — necessity holds directly.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"match_dropped": torch.tensor(1.0)},
        alternatives={"match_dropped": torch.tensor(0.0)},
        witnesses={"lightning": None},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a:
        with condition(data={**obs_both, **evidence_a}):
            posterior_a = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + ["__cause____antecedent_match_dropped", "forest_fire"],
            )()

logp_a = compute_logp_single(
    posterior_a, "match_dropped", "forest_fire", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_a = ac_check(logp_a)

# ── Query B: lightning is a cause (both=1 context) ───────────────────────────
# Match held at factual (1). Same logic as Query A by symmetry.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"lightning": torch.tensor(1.0)},
        alternatives={"lightning": torch.tensor(0.0)},
        witnesses={"match_dropped": None},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b:
        with condition(data={**obs_both, **evidence_b}):
            posterior_b = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + ["__cause____antecedent_lightning", "forest_fire"],
            )()

logp_b = compute_logp_single(
    posterior_b, "lightning", "forest_fire", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_b = ac_check(logp_b)

# ── Query C: joint {match, lightning} fails minimality (both=1 context) ──────
# The joint set satisfies necessity (removing both extinguishes AND to 0).
# But Queries A and B show each singleton is already a cause, so the joint
# set is not minimal — AC3 rejects it.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={
            "match_dropped": torch.tensor(1.0),
            "lightning": torch.tensor(1.0),
        },
        alternatives={
            "match_dropped": torch.tensor(0.0),
            "lightning": torch.tensor(0.0),
        },
        witnesses={},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_c:
        with condition(data={**obs_both, **evidence_c}):
            posterior_c = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + [
                        "__cause____antecedent_match_dropped",
                        "__cause____antecedent_lightning",
                        "forest_fire",
                    ],
            )()

logp_c, not_minimal_c = compute_logp_joint(
    posterior_c,
    ("match_dropped", "lightning"),
    "forest_fire",
    CONSEQUENT_SCALE,
    NUM_SAMPLES,
)
ac_c = ac_check(logp_c, not_minimal=not_minimal_c)

# ── Query D: match_dropped when lightning is absent (AC1 failure) ─────────────
# In this context, lightning=0 and forest_fire=0. The consequent (fire=1) did
# not occur, so AC1 is violated. In the ChiRho query the sufficiency check
# fails: with lightning=0, holding match at 1 cannot produce fire=1. The
# logp lands at the noise floor — a different failure mode from Query C's AC3.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s.supports,
        antecedents={"match_dropped": torch.tensor(1.0)},
        alternatives={"match_dropped": torch.tensor(0.0)},
        witnesses={"lightning": None},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_d:
        with condition(data={**obs_lightning_absent, **evidence_d}):
            posterior_d = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s.supports.keys())
                    + ["__cause____antecedent_match_dropped", "forest_fire"],
            )()

logp_d = compute_logp_single(
    posterior_d, "match_dropped", "forest_fire", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_d = ac_check(logp_d)

print(f"[Conjunctive] match_dropped is cause (both=1 context): {ac_a}")
print(f"[Conjunctive] lightning is cause (both=1 context):     {ac_b}")
print(f"[Conjunctive] Joint set not minimal:                   {ac_c}  # AC3 fails")
print(f"[Conjunctive] match_dropped, lightning=0 context:      {ac_d}  # AC1 fails — fire didn't happen")

assert ac_a, "Query A: match_dropped should be a cause in the both=1 context"
assert ac_b, "Query B: lightning should be a cause in the both=1 context"
assert not ac_c, "Query C: joint {match, lightning} should fail AC3 minimality"
assert not ac_d, "Query D: match should not be a cause when lightning=0 (fire didn't happen)"
print("\nAll assertions passed.")
