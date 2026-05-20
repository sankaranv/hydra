"""
Symmetric overdetermination: lightning and match-dropping (Halpern 2016 §2.3.1).

Lightning strikes and an arsonist drops a match simultaneously. Either alone
would have caused the forest fire. The fire results. This is the canonical
disjunctive overdetermination case.

Two models are compared in the same file:

`ff_disjunctive` (forest_fire = match OR lightning):
  - Singleton match_dropped fails AC2: lightning at 1 keeps the fire burning
    even when match is removed. No single antecedent is a but-for cause.
  - Joint {match, lightning} is the minimal actual cause: removing both
    together extinguishes the fire. Neither singleton works, so AC3 holds.

`ff_conjunctive` (forest_fire = match AND lightning):
  - Singleton match_dropped satisfies AC2: with lightning pinned at 1,
    removing match drops the product to 0. But-for holds directly.
  - Joint {match, lightning} violates AC3: each singleton is already
    sufficient, so the joint set is overcomplete.

The four-query pattern (A disjunctive singleton, B disjunctive joint,
C conjunctive singleton, D conjunctive joint) gives the full characterisation
of how OR vs AND structure changes what counts as a cause.
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


def ff_disjunctive():
    u_match_dropped = pyro.sample("u_match_dropped", dist.Bernoulli(0.5))
    u_lightning = pyro.sample("u_lightning", dist.Bernoulli(0.5))
    match_dropped = pyro.deterministic("match_dropped", u_match_dropped)
    lightning = pyro.deterministic("lightning", u_lightning)
    p_fire = torch.clamp(match_dropped + lightning, 0.0, 1.0)
    forest_fire = pyro.sample("forest_fire", dist.Bernoulli(p_fire))
    return {
        "match_dropped": match_dropped,
        "lightning": lightning,
        "forest_fire": forest_fire,
    }


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


observations = {
    "match_dropped": torch.tensor(1.0),
    "lightning": torch.tensor(1.0),
    "forest_fire": torch.tensor(1.0),
}

with ExtractSupports() as s_disj:
    condition(ff_disjunctive, data=observations)()

with ExtractSupports() as s_conj:
    condition(ff_conjunctive, data=observations)()

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

    World dims: dim 1 = first antecedent's split, dim 2 = second antecedent's
    split (preserving insertion order of the antecedents dict).

    AC3 check: when ant_b is preempted to factual (case=1), the factor still
    captures singleton-a's contribution because the b-dim world collapses to
    b's factual value. This lets us detect whether either singleton is already
    a sufficient cause without running separate queries.

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
    """True when necessity and sufficiency hold and the antecedent set is minimal."""
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    if not_minimal:
        print("  The antecedent set is not minimal.")
        return False
    return True


# ── Query A: singleton match_dropped, disjunctive ────────────────────────────
# Lightning is the witness, held at its factual value of 1. Removing match
# while lightning=1 leaves the fire burning (OR model) — necessity fails.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_disj.supports,
        antecedents={"match_dropped": torch.tensor(1.0)},
        alternatives={"match_dropped": torch.tensor(0.0)},
        witnesses={"lightning": None},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a:
        with condition(data={**observations, **evidence_a}):
            posterior_a = Predictive(
                ff_disjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_disj.supports.keys())
                    + ["__cause____antecedent_match_dropped", "forest_fire"],
            )()

logp_a = compute_logp_single(
    posterior_a, "match_dropped", "forest_fire", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_a = ac_check(logp_a)

# ── Query B: joint {match, lightning}, disjunctive ───────────────────────────
# No witness: the full joint set is the antecedent. Removing both together
# collapses OR to 0 → necessity holds. Neither singleton works alone →
# AC3 (minimality) satisfied → joint set is the minimal actual cause.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_disj.supports,
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
    ) as evidence_b:
        with condition(data={**observations, **evidence_b}):
            posterior_b = Predictive(
                ff_disjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_disj.supports.keys())
                    + [
                        "__cause____antecedent_match_dropped",
                        "__cause____antecedent_lightning",
                        "forest_fire",
                    ],
            )()

logp_b, not_minimal_b = compute_logp_joint(
    posterior_b,
    ("match_dropped", "lightning"),
    "forest_fire",
    CONSEQUENT_SCALE,
    NUM_SAMPLES,
)
ac_b = ac_check(logp_b, not_minimal=not_minimal_b)

# ── Query C: singleton match_dropped, conjunctive ────────────────────────────
# Lightning held at 1. With AND semantics, removing match drops the product
# from 1 to 0 — necessity holds directly, no joint set needed.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_conj.supports,
        antecedents={"match_dropped": torch.tensor(1.0)},
        alternatives={"match_dropped": torch.tensor(0.0)},
        witnesses={"lightning": None},
        consequents={"forest_fire": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_c:
        with condition(data={**observations, **evidence_c}):
            posterior_c = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_conj.supports.keys())
                    + ["__cause____antecedent_match_dropped", "forest_fire"],
            )()

logp_c = compute_logp_single(
    posterior_c, "match_dropped", "forest_fire", CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_c = ac_check(logp_c)

# ── Query D: joint {match, lightning}, conjunctive ───────────────────────────
# Each singleton already satisfies AC2 (either alone, with the other witness
# held at 1, makes the fire go out). The joint set's logp is above the noise
# floor, but the AC3 check detects that proper subsets are already causes.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_conj.supports,
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
    ) as evidence_d:
        with condition(data={**observations, **evidence_d}):
            posterior_d = Predictive(
                ff_conjunctive,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_conj.supports.keys())
                    + [
                        "__cause____antecedent_match_dropped",
                        "__cause____antecedent_lightning",
                        "forest_fire",
                    ],
            )()

logp_d, not_minimal_d = compute_logp_joint(
    posterior_d,
    ("match_dropped", "lightning"),
    "forest_fire",
    CONSEQUENT_SCALE,
    NUM_SAMPLES,
)
ac_d = ac_check(logp_d, not_minimal=not_minimal_d)

print(f"[Disjunctive] Singleton match_dropped: {ac_a}  # not an actual cause — overdetermination")
print(f"[Disjunctive] Joint {{match, lightning}}: {ac_b}  # actual cause — joint sufficiency")
print(f"[Conjunctive] Singleton match_dropped: {ac_c}  # actual cause — but-for holds")
print(f"[Conjunctive] Joint {{match, lightning}}: {ac_d}  # not minimal — fails AC3")

assert not ac_a, "Query A: singleton match in disjunctive should not be a cause"
assert ac_b, "Query B: joint {match, lightning} in disjunctive is the minimal cause"
assert ac_c, "Query C: singleton match in conjunctive should be a cause"
assert not ac_d, "Query D: joint set in conjunctive violates AC3 minimality"
print("\nAll assertions passed.")
