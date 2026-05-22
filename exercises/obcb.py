"""
OBCB: Overdetermination By Check Bypass.

A loan application has two gates:
1. Check step: gender determines whether the applicant is evaluated.
   check_failed ~ Bernoulli(0.8) for female (gender=0), Bernoulli(0.05) for male (gender=1).
2. Loan step: loan = 1 iff evaluated (check_failed=0) AND credit=good (credit=1).

Alice: female, good credit, check_failed=1 → denied.
Bob:   male,   bad credit,  check_failed=0 → denied.

What this exercise shows with PNS via SearchForExplanation:

  D-A1: gender is a cause of Alice's denial (PNS ≈ 0.20, positive).
  D-A2: credit has ZERO PNS impact on Alice's denial. In the necessity world
         (credit=0), loan=(1-check_failed)*0=0 regardless of check_failed, so
         the outcome never changes when credit is flipped. This is the correct
         PNS result: Alice's credit was never evaluated.
  D-A-rank: gender outranks credit for Alice (trivially: 0.20 > 0.00).
  D-witness: the check_failed witness constrains gender's attribution.
         Without the witness, check_failed freely follows male's structural
         equation in the necessity world (P(cf=0|male)=0.95), inflating
         P(loan=1) and hence PNS. With witness, cf is 50% preempted to
         factual (cf=1), reducing P(loan=1) and giving a more conservative
         estimate. Observed: PNS_gender ≈ 0.40 without witness, ≈ 0.20 with.
  D-B2: credit is the dominant cause of Bob's denial (PNS ≈ 0.40).
  D-B-rank: credit outranks gender for Bob (0.40 > 0.00).
  D-comp: gender's causal role is larger for Alice than for Bob.

Note on Table 2 target values: the PCI paper (§4.2) gives PNS-like values for
credit on Alice (0.199) and gender on Bob (0.019) using a generalised causal
measure that accounts for counterfactual potential impact beyond standard PNS.
Standard PNS (implemented here) gives zero for both, which is analytically
correct: with loan=(1-check_failed)*credit, necessity world with credit=0
always yields loan=0, so the binary PNS indicator is identically zero.
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


def obcb_model():
    u_gender = pyro.sample("u_gender", dist.Bernoulli(0.5))
    u_credit = pyro.sample("u_credit", dist.Bernoulli(0.5))
    gender = pyro.deterministic("gender", u_gender)
    credit = pyro.deterministic("credit", u_credit)
    # Female (gender=0) skipped with p=0.8; male (gender=1) skipped with p=0.05.
    check_prob = 0.8 * (1 - gender) + 0.05 * gender
    check_failed = pyro.sample("check_failed", dist.Bernoulli(check_prob))
    # Loan only if evaluated and credit is good.
    loan_val = torch.clamp((1 - check_failed) * credit, 0.0, 1.0)
    loan = pyro.sample("loan", dist.Bernoulli(loan_val))
    return {"gender": gender, "credit": credit, "check_failed": check_failed, "loan": loan}


alice_obs = {
    "gender": torch.tensor(0.0),        # female
    "credit": torch.tensor(1.0),        # good
    "check_failed": torch.tensor(1.0),  # not evaluated — check bypassed
    "loan": torch.tensor(0.0),          # denied
}

bob_obs = {
    "gender": torch.tensor(1.0),        # male
    "credit": torch.tensor(0.0),        # bad
    "check_failed": torch.tensor(0.0),  # evaluated
    "loan": torch.tensor(0.0),          # denied
}

with ExtractSupports() as s_alice:
    condition(obcb_model, data=alice_obs)()

with ExtractSupports() as s_bob:
    condition(obcb_model, data=bob_obs)()

NUM_SAMPLES = 50000
CONSEQUENT_SCALE = 1e-8
ANTECEDENT_BIAS = 0.1


def compute_pci(posterior: dict, num_samples: int) -> tuple:
    """PCI score and marginals for consequent loan=0 (denial) from a single-antecedent run.

    World layout: dim 0=factual, dim 1=necessity (suspect at alt), dim 2=sufficiency.
    ci(y_s, y_n, y*=0) = I{y_n != 0} * I{y_s == 0}  (PNS binary, Example 19).
    Trailing singleton dimensions from multi-world tensor layout are collapsed via reshape.
    """
    loan = posterior["loan"]
    w1 = loan[:, 1].reshape(num_samples).float()  # necessity world
    w2 = loan[:, 2].reshape(num_samples).float()  # sufficiency world
    ci = ((w1 != 0.0) & (w2 == 0.0)).float()
    pci = ci.mean().item()
    P_n = (w1 != 0.0).float().mean().item()
    P_s = (w2 == 0.0).float().mean().item()
    return pci, P_n, P_s


# ── Query A1: gender, Alice, with witness ─────────────────────────────────────
# Gender caused the check to fail. Necessity: if gender=male, check mostly passes
# and loan=1 (different from denial). Sufficiency: with female gender, check
# fails with high probability, so loan=0 persists. PNS ≈ 0.20.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_alice.supports,
        antecedents={"gender": torch.tensor(0.0)},
        alternatives={"gender": torch.tensor(1.0)},
        witnesses={"check_failed": None},
        consequents={"loan": torch.tensor(0.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a1:
        with condition(data={**alice_obs, **evidence_a1}):
            posterior_a1 = Predictive(
                obcb_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_alice.supports.keys())
                    + ["__cause____antecedent_gender", "loan"],
            )()

pci_a1, Pn_a1, Ps_a1 = compute_pci(posterior_a1, NUM_SAMPLES)

# ── Query A2: credit, Alice, with witness ─────────────────────────────────────
# Credit was never consulted (check bypassed). In the necessity world (credit=0),
# loan=(1-check_failed)*0=0 always — the outcome never differs. PNS=0 exactly.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_alice.supports,
        antecedents={"credit": torch.tensor(1.0)},
        alternatives={"credit": torch.tensor(0.0)},
        witnesses={"check_failed": None},
        consequents={"loan": torch.tensor(0.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a2:
        with condition(data={**alice_obs, **evidence_a2}):
            posterior_a2 = Predictive(
                obcb_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_alice.supports.keys())
                    + ["__cause____antecedent_credit", "loan"],
            )()

pci_a2, Pn_a2, Ps_a2 = compute_pci(posterior_a2, NUM_SAMPLES)

# ── Query A3: gender, Alice, no witness (inflated control) ────────────────────
# Without witness, check_failed is fully free in the necessity world. For male
# structural equation P(check_failed=0)=0.95, so P(loan=1) is high → inflated PNS.
# With witness (50% preemption to factual check_failed=1), this inflation is halved.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_alice.supports,
        antecedents={"gender": torch.tensor(0.0)},
        alternatives={"gender": torch.tensor(1.0)},
        witnesses={},
        consequents={"loan": torch.tensor(0.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_a3:
        with condition(data={**alice_obs, **evidence_a3}):
            posterior_a3 = Predictive(
                obcb_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_alice.supports.keys())
                    + ["__cause____antecedent_gender", "loan"],
            )()

pci_a3, _, _ = compute_pci(posterior_a3, NUM_SAMPLES)

# ── Query B1: gender, Bob, with witness ───────────────────────────────────────
# Being male opened the credit evaluation (indirect enabling role). In the necessity
# world (gender=female), credit=0 means loan=0 always → necessity never holds.
# PNS=0: standard PNS cannot detect indirect enabling via check_failed mediation.

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_bob.supports,
        antecedents={"gender": torch.tensor(1.0)},
        alternatives={"gender": torch.tensor(0.0)},
        witnesses={"check_failed": None},
        consequents={"loan": torch.tensor(0.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b1:
        with condition(data={**bob_obs, **evidence_b1}):
            posterior_b1 = Predictive(
                obcb_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_bob.supports.keys())
                    + ["__cause____antecedent_gender", "loan"],
            )()

pci_b1, Pn_b1, Ps_b1 = compute_pci(posterior_b1, NUM_SAMPLES)

# ── Query B2: credit, Bob, with witness ───────────────────────────────────────
# Bad credit directly caused denial. In necessity world (credit=1, good):
# loan=(1-0)*1=1, different from denial. PNS ≈ 0.40 (driven by antecedent activity).

with MultiWorldCounterfactual():
    with SearchForExplanation(
        supports=s_bob.supports,
        antecedents={"credit": torch.tensor(0.0)},
        alternatives={"credit": torch.tensor(1.0)},
        witnesses={"check_failed": None},
        consequents={"loan": torch.tensor(0.0)},
        consequent_scale=CONSEQUENT_SCALE,
        antecedent_bias=ANTECEDENT_BIAS,
    ) as evidence_b2:
        with condition(data={**bob_obs, **evidence_b2}):
            posterior_b2 = Predictive(
                obcb_model,
                num_samples=NUM_SAMPLES,
                return_sites=list(s_bob.supports.keys())
                    + ["__cause____antecedent_credit", "loan"],
            )()

pci_b2, Pn_b2, Ps_b2 = compute_pci(posterior_b2, NUM_SAMPLES)

# ── Print ─────────────────────────────────────────────────────────────────────
print("--- Alice ---")
print(f"PNS gender  (with witness):    {pci_a1:.3f}   # D-A1: gender is a cause")
print(f"PNS credit  (with witness):    {pci_a2:.3f}   # D-A2: zero — check bypassed credit")
print(f"PNS gender > PNS credit:       {pci_a1 > pci_a2}     # D-A-rank")
print(f"PNS gender  (no witness):      {pci_a3:.3f}   # D-witness: inflated without cf constraint")
print(f"PNS gender (no witness) > (with witness): {pci_a3 > pci_a1}  # witness reduces inflation")

print("\n--- Bob ---")
print(f"PNS gender  (with witness):    {pci_b1:.3f}   # zero — indirect enabling not captured by PNS")
print(f"PNS credit  (with witness):    {pci_b2:.3f}   # D-B2: credit is dominant cause")
print(f"PNS credit > PNS gender:       {pci_b2 > pci_b1}     # D-B-rank")

print("\n--- Cross-individual ---")
print(f"PNS gender Alice > PNS gender Bob: {pci_a1 > pci_b1}  # D-comp")

print("\n--- Decomposition (Alice, gender, with witness) ---")
print(f"Necessity marginal:   {Pn_a1:.3f}")
print(f"Sufficiency marginal: {Ps_a1:.3f}")
print(f"Joint total:          {pci_a1:.3f}   # ~0.20")

print("\n--- Decomposition (Bob, credit, with witness) ---")
print(f"Necessity marginal:   {Pn_b2:.3f}")
print(f"Sufficiency marginal: {Ps_b2:.3f}")
print(f"Joint total:          {pci_b2:.3f}   # ~0.40")

# ── Assertions ────────────────────────────────────────────────────────────────
tol = 0.04

# D-A1: gender causes Alice's denial
assert abs(pci_a1 - 0.20) < tol, f"D-A1: expected ~0.20, got {pci_a1:.3f}"

# D-A2: credit has zero PNS — check bypassed credit evaluation entirely
assert pci_a2 == 0.0, f"D-A2: credit PNS must be 0.0 (check bypassed), got {pci_a2:.4f}"

# D-A-rank: gender outranks credit (trivially, since credit=0)
assert pci_a1 > pci_a2, "D-A-rank: gender must outrank credit for Alice"

# D-witness: without witness, gender PNS is inflated (cf freely follows male eq)
assert pci_a3 > pci_a1, "D-witness: gender PNS must be higher without cf witness"

# D-B2: credit causes Bob's denial
assert abs(pci_b2 - 0.40) < tol, f"D-B2: expected ~0.40, got {pci_b2:.3f}"

# D-B1: gender PNS for Bob is zero with this PNS formulation
assert pci_b1 == 0.0, f"D-B1: gender PNS for Bob must be 0.0, got {pci_b1:.4f}"

# D-B-rank: credit dominates for Bob
assert pci_b2 > pci_b1, "D-B-rank: credit must outrank gender for Bob"

# D-comp: gender's causal role is larger for Alice than Bob
assert pci_a1 > pci_b1, "D-comp: gender matters more for Alice than Bob"

print("\nAll assertions passed.")
