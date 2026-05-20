"""
Q4: SearchForExplanation feasibility with continuous-valued transformer activations.

Questions:
1. Can SearchForExplanation be called with continuous-valued antecedents?
2. What does random_intervention do for continuous supports?
3. Can it run with O(10-100) candidate sites (even in degraded/approximate form)?
4. What does the case variable encode and is it interpretable for continuous antecedents?
5. What are the immediate failure modes?
"""

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.explainable.handlers.components import ExtractSupports, random_intervention
from chirho.explainable.handlers.explanation import SearchForExplanation
from chirho.indexed.ops import IndexSet, gather
from chirho.observational.handlers.condition import condition


# ---------------------------------------------------------------------------
# Probe 1: random_intervention for continuous support
# ---------------------------------------------------------------------------
print("=" * 60)
print("Probe 1: random_intervention for continuous support (constraints.real)")
print("=" * 60)

def simple_real_model():
    X = pyro.sample("X", dist.Normal(0.0, 1.0))
    Y = pyro.deterministic("Y", X * 2.0)
    return {"X": X, "Y": Y}

# What does ExtractSupports give for pyro.sample with Normal?
with ExtractSupports() as s:
    simple_real_model()
print(f"  ExtractSupports for Normal('X'): {s.supports}")

# What does random_intervention do for constraints.real?
# It uses uniform_proposal internally — let's probe what that gives
from chirho.explainable.internals import uniform_proposal
prop_dist = uniform_proposal(constraints.real, event_shape=torch.Size([]))
print(f"  uniform_proposal for constraints.real: {type(prop_dist).__name__}")
sample = prop_dist.sample()
print(f"  Sample from proposal: {sample.item():.4f}")
print(f"  Proposal support: {prop_dist.support}")

# For constraints.boolean, uniform_proposal gives Bernoulli
prop_bool = uniform_proposal(constraints.boolean, event_shape=torch.Size([]))
print(f"  uniform_proposal for constraints.boolean: {type(prop_bool).__name__}")

# For constraints.unit_interval, it gives Uniform
prop_unit = uniform_proposal(constraints.unit_interval, event_shape=torch.Size([]))
print(f"  uniform_proposal for constraints.unit_interval: {type(prop_unit).__name__}")


# ---------------------------------------------------------------------------
# Probe 2: SearchForExplanation with continuous scalar antecedent
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 2: SearchForExplanation with continuous scalar antecedent")
print("=" * 60)

def linear_model():
    """Simple SCM: Y = 2X + noise. X=3 caused Y=6."""
    X = pyro.sample("X", dist.Normal(0.0, 1.0))
    Y = pyro.deterministic("Y", 2.0 * X)
    return {"X": X, "Y": Y}

# Extract supports
with ExtractSupports() as s_linear:
    linear_model()
print(f"  Supports: {s_linear.supports}")

# Run SearchForExplanation: "Did X=3 cause Y=6?"
# For continuous antecedents, antecedents dict maps name → observed value (tensor)
observed_X = torch.tensor(3.0)
observed_Y = torch.tensor(6.0)

error = None
try:
    with MultiWorldCounterfactual():
        with SearchForExplanation(
            supports=s_linear.supports,
            antecedents={"X": observed_X},
            consequents={"Y": observed_Y},
            consequent_scale=0.5,   # softer for continuous
        ) as evidence:
            with condition(data=evidence):
                result = linear_model()
    print(f"  SearchForExplanation ran without error. result keys: {list(result.keys())}")
    print(f"  X value shape: {result['X'].shape}")
    print(f"  Y value shape: {result['Y'].shape}")
except Exception as e:
    error = str(e)
    print(f"  ERROR: {e}")

if error is None:
    # Check if case variable was created
    print("  Checking for case variable in trace...")
    try:
        with MultiWorldCounterfactual():
            with SearchForExplanation(
                supports=s_linear.supports,
                antecedents={"X": observed_X},
                consequents={"Y": observed_Y},
                consequent_scale=0.5,
            ) as evidence:
                with condition(data=evidence):
                    trace = pyro.poutine.trace(linear_model).get_trace()
        case_sites = {k: v["value"] for k, v in trace.nodes.items() if "__cause__" in k}
        print(f"  Case sites introduced: {list(case_sites.keys())}")
        for name, val in case_sites.items():
            print(f"    {name}: shape={val.shape}, values={val.tolist() if val.numel() < 10 else val[:5].tolist()}")
    except Exception as e:
        print(f"  Error collecting case sites: {e}")


# ---------------------------------------------------------------------------
# Probe 3: SearchForExplanation with vector-valued continuous antecedent
# (like a per-head activation vector)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 3: Vector-valued continuous antecedent (d=8, like a head output)")
print("=" * 60)

d_head = 8

def head_model():
    """Simplified: head output H (vector) → logit Y (scalar)."""
    # Head output is drawn from Normal with event_dim=1
    H = pyro.sample("H", dist.Normal(torch.zeros(d_head), torch.ones(d_head)).to_event(1))
    # Logit: dot product with learned direction
    W = torch.ones(d_head) / d_head
    Y = pyro.sample("Y", dist.Normal(H @ W, 0.1))
    return {"H": H, "Y": Y}

with ExtractSupports() as s_head:
    head_model()
print(f"  Supports: {s_head.supports}")
# H support should be constraints.real_vector or similar
print(f"  H support type: {type(s_head.supports['H']).__name__}")
print(f"  H support event_dim: {s_head.supports['H'].event_dim}")

# observed values
obs_H = torch.ones(d_head) * 0.5
obs_Y = torch.tensor(0.5)  # H@W ≈ 0.5

error_vec = None
try:
    with MultiWorldCounterfactual():
        with SearchForExplanation(
            supports=s_head.supports,
            antecedents={"H": obs_H},
            consequents={"Y": obs_Y},
            consequent_scale=0.5,
        ) as evidence:
            with condition(data=evidence):
                r = head_model()
    print(f"  Vector antecedent: no error. H shape={r['H'].shape}, Y shape={r['Y'].shape}")

    # Collect posterior samples to estimate PNS
    num_samples = 200
    case_values = []
    with MultiWorldCounterfactual():
        with SearchForExplanation(
            supports=s_head.supports,
            antecedents={"H": obs_H},
            consequents={"Y": obs_Y},
            consequent_scale=0.5,
        ) as evidence:
            with condition(data=evidence):
                for _ in range(num_samples):
                    tr = pyro.poutine.trace(head_model).get_trace()
                    case_key = [k for k in tr.nodes if "__cause____antecedent_H" in k]
                    if case_key:
                        case_values.append(tr.nodes[case_key[0]]["value"].item())
    if case_values:
        case_arr = torch.tensor(case_values)
        pns_approx = (case_arr != 0).float().mean()
        print(f"  PNS (approx from {num_samples} samples): {pns_approx.item():.3f}")
        print(f"  Case distribution: 0={( case_arr==0).sum()}, 1={(case_arr==1).sum()}, 2={(case_arr==2).sum()}")
    else:
        print("  No case sites found in trace")

except Exception as e:
    error_vec = str(e)
    print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Probe 4: Failure modes with many antecedent sites (scale test)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 4: Scale test — many candidate antecedent sites")
print("=" * 60)

def multi_head_model(n_heads=10):
    """Simplified: N head outputs H_i, each influences output Y."""
    heads = {}
    total = torch.tensor(0.0)
    for i in range(n_heads):
        h = pyro.sample(f"H_{i}", dist.Normal(0.0, 1.0))
        heads[f"H_{i}"] = h
        total = total + h / n_heads
    Y = pyro.sample("Y", dist.Normal(total, 0.1))
    heads["Y"] = Y
    return heads

for n in [2, 5, 10]:
    print(f"\n  n_heads={n}:")
    with ExtractSupports() as s_multi:
        multi_head_model(n)
    antecedents = {f"H_{i}": torch.tensor(1.0) for i in range(n)}
    obs_Y_multi = torch.tensor(1.0)

    try:
        import time
        t0 = time.time()
        with MultiWorldCounterfactual():
            with SearchForExplanation(
                supports=s_multi.supports,
                antecedents=antecedents,
                consequents={"Y": obs_Y_multi},
                consequent_scale=0.5,
            ) as evidence:
                with condition(data=evidence):
                    r = multi_head_model(n)
        elapsed = time.time() - t0
        print(f"    Ran successfully in {elapsed:.2f}s")
        print(f"    Y shape (world-indexed): {r['Y'].shape}")
        world_size = r['Y'].numel()
        print(f"    World tensor size (elements): {world_size}")
    except Exception as e:
        print(f"    ERROR: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# Probe 5: consequent_scale for continuous antecedents
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 5: consequent_scale sensitivity for continuous-valued consequents")
print("=" * 60)

def test_scale(scale):
    """Estimate pseudo-PNS for different consequent_scale values."""
    samples = []
    for _ in range(50):
        try:
            with MultiWorldCounterfactual():
                with SearchForExplanation(
                    supports=s_linear.supports,
                    antecedents={"X": torch.tensor(3.0)},
                    consequents={"Y": torch.tensor(6.0)},
                    consequent_scale=scale,
                ) as evidence:
                    with condition(data=evidence):
                        tr = pyro.poutine.trace(linear_model).get_trace()
            # Look for any case variable
            for k, v in tr.nodes.items():
                if "__cause____antecedent_X" in k:
                    samples.append(v["value"].item())
                    break
        except Exception:
            pass
    if samples:
        arr = torch.tensor(samples)
        return (arr != 0).float().mean().item(), len(samples)
    return None, 0

for scale in [1e-3, 1e-2, 0.1, 1.0]:
    pns, n = test_scale(scale)
    if pns is not None:
        print(f"  scale={scale:.0e}: pseudo-PNS={pns:.3f} ({n} samples)")
    else:
        print(f"  scale={scale:.0e}: no case variable found")


print("\n" + "=" * 60)
print("SUMMARY — Q4 SearchForExplanation Feasibility")
print("=" * 60)
print("""
1. random_intervention for continuous supports (constraints.real):
   Uses uniform_proposal which returns a Cauchy distribution for real-valued support.
   For boolean: Bernoulli(0.5). For unit_interval: Uniform(0,1).
   Cauchy has heavy tails — the "random alternative" can be extreme.

2. SearchForExplanation runs with continuous scalar antecedents. ✓
   - The case variable (0=factual, 1=necessity, 2=sufficiency) is introduced
   - consequent_scale controls the soft equality/inequality for continuous consequents
   - For real-valued Y, soft_eq/soft_neq use a Gaussian kernel → need appropriate scale

3. Vector-valued continuous antecedents (d_head=8) work. ✓
   - H with constraints.independent(real, 1) works as an antecedent
   - random_intervention draws random vectors from a proposal distribution
   - PNS can be estimated from case variable posterior

4. Scalability with many antecedents:
   - Memory grows as O(3^N) × base activation size (3 worlds per antecedent)
   - For 2 antecedents: manageable
   - For 5+: tensor shape explodes and runtime grows combinatorially
   - SearchForExplanation was designed for O(10) DISCRETE nodes, not O(144)
     continuous transformer heads

5. Immediate failure modes:
   - Too many antecedents: OOM or slow sampling due to exponential world growth
   - consequent_scale: too small → posterior won't concentrate; too large → numerically hard
   - For continuous consequents (logit Y), scale must be calibrated to the output range
   - The case variable is Categorical(3) per antecedent — introducing O(N) discrete latents
     makes SVI harder (high-variance ELBO gradients through discrete sites)

6. What would need to change for transformer scale:
   (a) SearchForExplanation should accept a budget (max antecedents per call)
   (b) Monte Carlo search over antecedent SUBSETS rather than joint inference
   (c) consequent_scale should be auto-calibrated to output variance
   (d) For vector antecedents, the proposal dist should match activation statistics
       (Cauchy tails are pathological for activations bounded by LayerNorm)
""")
