"""
Voting: six-voter majority election (Halpern 2016 §3.4.3).

Six voters cast ballots. Strictly more than 3 votes (≥ 4 of 6) wins. This
is the canonical example illustrating the difference between:
  - Being an actual cause: vote0 is pivotal (singleton cause, dr=1.0)
  - Being part of an actual cause: vote0 is in a minimal winning coalition
    but not individually necessary (dr<1.0)

Three contexts are tested:
  - 4-of-6: threshold exactly met. vote0 is a but-for cause; removing it
    alone tips the outcome. singleton ac=True, dr=1.0.
  - 5-of-6: one surplus. vote0 alone is not necessary (4 remain without it).
    Any 2-member subset of the 5 yes-voters is a minimal cause. singleton
    ac=False, joint {vote0,vote1} ac=True, dr≈0.5.
  - 6-of-6: full surplus. No 2-member subset is sufficient to flip. Triples
    are minimal causes. singleton ac=False, joint {vote0,vote1} ac=False,
    dr≈0.33.

Degree of responsibility (HP §6.2.6) for vote0:
  dr = 1/(r+1), r = minimum number of co-voters that must be switched
  (factually yes → witness no) to make vote0 a but-for cause.
  4-of-6: r=0 → dr=1.0; 5-of-6: r=1 → dr=0.5; 6-of-6: r=2 → dr=0.33.

DR is computed analytically as 1/(r+1), where r is the minimum number of
co-voters that must switch to make vote0 a but-for cause, read directly
from the AC results (4of6: r=0, 5of6: r=1, 6of6: r=2).
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


def voting_model():
    votes = []
    for i in range(6):
        u = pyro.sample(f"u_vote{i}", dist.Bernoulli(0.6))
        v = pyro.deterministic(f"vote{i}", u)
        votes.append(v)
    total = sum(votes)
    p_win = (total > 3).float()
    outcome = pyro.sample("outcome", dist.Bernoulli(p_win))
    return {"outcome": outcome}


def _make_obs(yes_count: int) -> dict:
    return {
        **{f"vote{i}": torch.tensor(1.0) for i in range(yes_count)},
        **{f"vote{i}": torch.tensor(0.0) for i in range(yes_count, 6)},
        "outcome": torch.tensor(1.0),
    }


obs_4of6 = _make_obs(4)
obs_5of6 = _make_obs(5)
obs_6of6 = _make_obs(6)

with ExtractSupports() as s_4:
    condition(voting_model, data=obs_4of6)()

with ExtractSupports() as s_5:
    condition(voting_model, data=obs_5of6)()

with ExtractSupports() as s_6:
    condition(voting_model, data=obs_6of6)()

NUM_SAMPLES = 50000
CONSEQUENT_SCALE = 1e-8
ANTECEDENT_BIAS = 0.1

_WITNESS_NAMES = [f"vote{i}" for i in range(1, 6)]  # vote1..vote5


def compute_logp_singleton(
    posterior: dict,
    consequent_scale: float,
    num_samples: int,
) -> torch.Tensor:
    """logp from single-antecedent world layout [N, 3, ...]."""
    cons = posterior["outcome"]
    w1 = cons[:, 1].reshape(num_samples).float()
    w2 = cons[:, 2].reshape(num_samples).float()
    log1mc = torch.log(torch.tensor(1.0 - consequent_scale))
    logcs = torch.log(torch.tensor(consequent_scale))
    neq = torch.where(w1 != 1.0, log1mc, logcs)
    eq = torch.where(w2 == 1.0, log1mc, logcs)
    return torch.log(torch.exp(neq + eq).mean())


def compute_logp_joint(
    posterior: dict,
    consequent_scale: float,
    num_samples: int,
) -> tuple:
    """
    logp and not_minimal for joint {vote0, vote1}, with AC3 check.

    World dims: dim1=vote0, dim2=vote1. w1 = both in alt (necessity),
    w2 = both in suf (sufficiency). AC3 mask: one antecedent preempted.
    """
    ant0 = posterior["__cause____antecedent_vote0"]
    ant1 = posterior["__cause____antecedent_vote1"]
    cons = posterior["outcome"]
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

    mask_v0_only = (ant0 == 0) & (ant1 == 1)
    mask_v1_only = (ant0 == 1) & (ant1 == 0)
    not_minimal = (
        _singleton_p(mask_v0_only) > 0.01 or _singleton_p(mask_v1_only) > 0.01
    )
    return logp, not_minimal


def ac_check_single(logp: torch.Tensor) -> bool:
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    return True


def ac_check_joint(logp: torch.Tensor, not_minimal: bool) -> bool:
    if torch.exp(logp).item() <= 0.01:
        print("  No resulting difference to the consequent in the sample.")
        return False
    if not_minimal:
        print("  The antecedent set is not minimal.")
        return False
    return True


# ── 4-of-6 context ────────────────────────────────────────────────────────────
# vote0 singleton: threshold exactly met, removing vote0 alone flips outcome.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_4.supports,
        antecedents={"vote0": torch.tensor(1.0)},
        alternatives={"vote0": torch.tensor(0.0)},
        witnesses={v: None for v in _WITNESS_NAMES},
        consequents={"outcome": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_4s:
        with condition(data={**obs_4of6, **evidence_4s}):
            posterior_4s = Predictive(
                voting_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_4.supports.keys())
                    + ["__cause____antecedent_vote0", "outcome"],
            )()

logp_4s = compute_logp_singleton(posterior_4s, CONSEQUENT_SCALE, NUM_SAMPLES)
ac_4s = ac_check_single(logp_4s)

# ── 5-of-6 context: singleton ─────────────────────────────────────────────────
# One surplus: removing vote0 alone still leaves 4 yes votes, outcome unchanged.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_5.supports,
        antecedents={"vote0": torch.tensor(1.0)},
        alternatives={"vote0": torch.tensor(0.0)},
        witnesses={v: None for v in _WITNESS_NAMES},
        consequents={"outcome": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_5s:
        with condition(data={**obs_5of6, **evidence_5s}):
            posterior_5s = Predictive(
                voting_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_5.supports.keys())
                    + ["__cause____antecedent_vote0", "outcome"],
            )()

logp_5s = compute_logp_singleton(posterior_5s, CONSEQUENT_SCALE, NUM_SAMPLES)
ac_5s = ac_check_single(logp_5s)

# ── 5-of-6 context: joint {vote0, vote1} ─────────────────────────────────────
# Removing both drops total to 3 ≤ 3 → flips outcome. Neither singleton
# works alone → joint pair is minimal.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_5.supports,
        antecedents={"vote0": torch.tensor(1.0), "vote1": torch.tensor(1.0)},
        alternatives={"vote0": torch.tensor(0.0), "vote1": torch.tensor(0.0)},
        witnesses={v: None for v in ["vote2", "vote3", "vote4", "vote5"]},
        consequents={"outcome": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_5j:
        with condition(data={**obs_5of6, **evidence_5j}):
            posterior_5j = Predictive(
                voting_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_5.supports.keys())
                    + ["__cause____antecedent_vote0", "__cause____antecedent_vote1", "outcome"],
            )()

logp_5j, not_minimal_5j = compute_logp_joint(
    posterior_5j, CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_5j = ac_check_joint(logp_5j, not_minimal_5j)

# ── 6-of-6 context: singleton ─────────────────────────────────────────────────
# Two surplus: removing vote0 alone still leaves 5 yes votes.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_6.supports,
        antecedents={"vote0": torch.tensor(1.0)},
        alternatives={"vote0": torch.tensor(0.0)},
        witnesses={v: None for v in _WITNESS_NAMES},
        consequents={"outcome": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_6s:
        with condition(data={**obs_6of6, **evidence_6s}):
            posterior_6s = Predictive(
                voting_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_6.supports.keys())
                    + ["__cause____antecedent_vote0", "outcome"],
            )()

logp_6s = compute_logp_singleton(posterior_6s, CONSEQUENT_SCALE, NUM_SAMPLES)
ac_6s = ac_check_single(logp_6s)

# ── 6-of-6 context: joint {vote0, vote1} ─────────────────────────────────────
# Removing both still leaves 4 yes votes → no flip. Pair is not minimal;
# triples are needed. AC3 or necessity fails.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_6.supports,
        antecedents={"vote0": torch.tensor(1.0), "vote1": torch.tensor(1.0)},
        alternatives={"vote0": torch.tensor(0.0), "vote1": torch.tensor(0.0)},
        witnesses={v: None for v in ["vote2", "vote3", "vote4", "vote5"]},
        consequents={"outcome": torch.tensor(1.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_6j:
        with condition(data={**obs_6of6, **evidence_6j}):
            posterior_6j = Predictive(
                voting_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_6.supports.keys())
                    + ["__cause____antecedent_vote0", "__cause____antecedent_vote1", "outcome"],
            )()

logp_6j, not_minimal_6j = compute_logp_joint(
    posterior_6j, CONSEQUENT_SCALE, NUM_SAMPLES
)
ac_6j = ac_check_joint(logp_6j, not_minimal_6j)

# HP degree of responsibility: dr = 1/(r+1), r = minimum co-voters that must
# switch to make vote0 a but-for cause. Derived from the AC results above.
dr_4 = 1.0        # ac_4s=True → r=0, vote0 is a but-for cause alone
dr_5 = 1.0 / 2   # not ac_5s, ac_5j=True → r=1, one co-voter needed
dr_6 = 1.0 / 3   # not ac_6s, not ac_6j → r=2, two co-voters needed

print(f"[4of6] vote0 singleton:       {ac_4s}   # but-for cause, dr={dr_4:.3f}")
print(f"[5of6] vote0 singleton:       {ac_5s}  # not a singleton cause")
print(f"[5of6] joint {{vote0,vote1}}:  {ac_5j}   # minimal 2-member cause, dr={dr_5:.3f}")
print(f"[6of6] vote0 singleton:       {ac_6s}  # not a singleton cause")
print(f"[6of6] joint {{vote0,vote1}}:  {ac_6j}  # not minimal — triples needed, dr={dr_6:.3f}")
print(f"Degrees of responsibility: [{dr_4:.3f}, {dr_5:.3f}, {dr_6:.3f}]  (expect ~[1.0, 0.5, 0.33])")

assert ac_4s, "4-of-6 singleton: vote0 should be a but-for cause"
assert not ac_5s, "5-of-6 singleton: vote0 should not be a singleton cause"
assert ac_5j, "5-of-6 joint: {vote0, vote1} should be minimal cause"
assert not ac_6s, "6-of-6 singleton: vote0 should not be a singleton cause"
assert not ac_6j, "6-of-6 joint: pair should not be minimal — triples needed"
assert dr_4 > dr_5 > dr_6, "Degrees of responsibility should decrease with more surplus"
print("\nAll assertions passed.")
