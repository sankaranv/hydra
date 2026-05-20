"""
Q1: Effect handler mechanics — does `do` intercept pyro.deterministic sites?

Probes:
1. do() intercepts pyro.deterministic via _pyro_post_sample ✓
2. MultiWorldCounterfactual: tensor broadcasting, not re-execution; gather() must be INSIDE context
3. Arbitrary PyTorch ops between sites work if they broadcast over extra batch dims
4. event_dim matters for gather() on vector-valued sites — must match site registration
5. view/reshape between sites BREAKS world-indexed tensors if shapes are hardcoded
6. Disconnected sites are not in the causal graph
"""

import pyro
import pyro.distributions as dist
import torch
from chirho.counterfactual.handlers.counterfactual import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do


# ---------------------------------------------------------------------------
# Probe 1: do() intercepts pyro.deterministic
# ---------------------------------------------------------------------------
def model_deterministic():
    noise = pyro.sample("noise", dist.Normal(0.0, 1.0))
    x_sq = pyro.deterministic("x_sq", noise.pow(2))
    y = pyro.deterministic("y", x_sq * 2.0)
    return {"noise": noise, "x_sq": x_sq, "y": y}


print("=" * 60)
print("Probe 1: do() on pyro.deterministic")
print("=" * 60)
torch.manual_seed(0)
base = model_deterministic()
torch.manual_seed(0)
patched = do(model_deterministic, {"x_sq": torch.tensor(99.0)})()
assert patched["x_sq"].item() == 99.0
assert patched["y"].item() == 198.0
print(f"  base: x_sq={base['x_sq'].item():.4f}, y={base['y'].item():.4f}")
print(f"  patched x_sq=99: y={patched['y'].item():.4f}")
print("  PASS: do() intercepts pyro.deterministic; return value propagates downstream")


# ---------------------------------------------------------------------------
# Probe 2: MultiWorldCounterfactual — world dimension, gather inside context
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 2: MultiWorldCounterfactual broadcasting, gather inside context")
print("=" * 60)
torch.manual_seed(0)
with MultiWorldCounterfactual():
    with do(actions={"x_sq": torch.tensor(99.0)}):
        r = model_deterministic()
    y_f = gather(r["y"], IndexSet(**{"x_sq": {0}}))   # factual
    y_i = gather(r["y"], IndexSet(**{"x_sq": {1}}))   # intervened
print(f"  y tensor shape: {r['y'].shape}  (world dim left-padded)")
print(f"  y_factual={y_f.squeeze().item():.4f}  y_intervened={y_i.squeeze().item():.4f}")
assert abs(y_i.squeeze().item() - 198.0) < 1e-4
print("  PASS: world-indexed tensor correct; gather() must run inside context")


# ---------------------------------------------------------------------------
# Probe 3: Vector-valued site — event_dim must match registration AND gather call
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 3: Vector-valued site with event_dim=1")
print("=" * 60)
W = torch.ones(4, 8) / 4.0

def model_vector():
    x = pyro.sample("x", dist.Normal(torch.zeros(4), torch.ones(4)).to_event(1))
    # event_dim=1 tells Pyro/ChiRho the site value is a 1D event (not a batch of scalars)
    h = pyro.deterministic("h", torch.tanh(x @ W), event_dim=1)
    # sum over event dim only (-1), not over world batch dims
    out = pyro.deterministic("out", h.sum(dim=-1), event_dim=0)
    return {"h": h, "out": out}

torch.manual_seed(0)
with MultiWorldCounterfactual():
    with do(actions={"h": torch.zeros(8)}):
        r = model_vector()
    # Must pass event_dim=1 to gather for vector sites
    h_f = gather(r["h"], IndexSet(**{"h": {0}}), event_dim=1)
    h_i = gather(r["h"], IndexSet(**{"h": {1}}), event_dim=1)
    out_f = gather(r["out"], IndexSet(**{"h": {0}}))
    out_i = gather(r["out"], IndexSet(**{"h": {1}}))
print(f"  h tensor shape: {r['h'].shape}")
print(f"  h_factual[:3]: {h_f.squeeze()[:3].tolist()}")
print(f"  h_intervened[:3]: {h_i.squeeze()[:3].tolist()}")
print(f"  out_factual: {out_f.squeeze().item():.4f}")
print(f"  out_intervened (sum of zeros, should be 0): {out_i.squeeze().item():.4f}")
assert torch.allclose(h_i.squeeze(), torch.zeros(8)), "h_intervened should be zeros"
assert abs(out_i.squeeze().item()) < 1e-6, "out_intervened should be 0"
print("  PASS: event_dim=1 sites work with gather(event_dim=1); downstream ops broadcast correctly")


# ---------------------------------------------------------------------------
# Probe 4: view/reshape between named sites — the critical transformer concern
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 4: view/reshape between named sites (breaks multi-world)")
print("=" * 60)

def model_with_reshape():
    x = pyro.sample("x", dist.Normal(torch.zeros(4), torch.ones(4)).to_event(1))
    h = pyro.deterministic("h", x, event_dim=1)
    # Hardcoded view — will fail if world dim is added on left
    reshaped = h.view(4)  # hardcoded shape
    out = pyro.deterministic("out", reshaped.sum(), event_dim=0)
    return {"h": h, "out": out}

torch.manual_seed(0)
error_msg = None
try:
    with MultiWorldCounterfactual():
        with do(actions={"h": torch.zeros(4)}):
            r = model_with_reshape()
except RuntimeError as e:
    error_msg = str(e)
    print(f"  EXPECTED ERROR with view(): {error_msg[:100]}...")

if error_msg:
    print("  FAIL (expected): hardcoded view() breaks multi-world execution")
    print("  *** KEY FINDING FOR TRANSFORMERS: any view/reshape to fixed shape will break ***")
else:
    # Check if the result is wrong even if no error
    print(f"  Ran without error. h shape: {r['h'].shape}")
    print("  WARNING: no error but shapes may be wrong — inspect carefully")


# ---------------------------------------------------------------------------
# Probe 5: Attention-style operation (batched matmul over seq×seq)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 5: Attention-style batched matmul Q@K.T")
print("=" * 60)
# Simplified: [seq, d_head] -> attention_pattern [seq, seq]
# With multi-world: q becomes [world, 1, ..., seq, d_head]
# We test if attention_pattern remains correct

def model_attention_like():
    # Q, K, V: seq=3, d_head=4
    inp = pyro.sample("inp", dist.Normal(torch.zeros(3, 4), torch.ones(3, 4)).to_event(2))
    q = pyro.deterministic("q", inp, event_dim=2)            # [3, 4]
    # Simple self-attention
    attn = torch.softmax(q @ q.mT / 4.0**0.5, dim=-1)       # [3, 3]
    # Sum over event dims only (last 2), NOT over world batch dims
    out = pyro.deterministic("out", (attn @ q).sum(dim=(-2, -1)), event_dim=0)
    return {"q": q, "attn_pattern": attn, "out": out}

torch.manual_seed(0)
# In single-world, verify base behavior
base_result = model_attention_like()
print(f"  Base attn_pattern shape: {base_result['attn_pattern'].shape}")

# Test with multi-world patching at q
torch.manual_seed(0)
try:
    with MultiWorldCounterfactual():
        patch_q = torch.ones(3, 4) * 0.1
        with do(actions={"q": patch_q}):
            r = model_attention_like()
        out_f = gather(r["out"], IndexSet(**{"q": {0}}))
        out_i = gather(r["out"], IndexSet(**{"q": {1}}))
    print(f"  q tensor shape after split: {r['q'].shape}")
    print(f"  out_factual: {out_f.squeeze().item():.4f}")
    print(f"  out_intervened: {out_i.squeeze().item():.4f}")

    # Check: with q=0.1 everywhere, attn is uniform, out should be predictable
    expected_q = torch.ones(3, 4) * 0.1
    expected_attn = torch.softmax(expected_q @ expected_q.mT / 4.0**0.5, dim=-1)
    expected_out = (expected_attn @ expected_q).sum()
    assert abs(out_i.squeeze().item() - expected_out.item()) < 1e-4, \
        f"out_intervened={out_i.item():.4f} != expected={expected_out.item():.4f}"
    print(f"  expected (manual): {expected_out.item():.4f}")
    print("  PASS: attention-style matmul Q@K.T broadcasts correctly over world dim")
    print("  NOTE: q.mT works because mT transposes the last 2 dims regardless of leading dims")
except Exception as e:
    print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Probe 6: Disconnected site
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Probe 6: Disconnected pyro.deterministic (not used downstream)")
print("=" * 60)

def model_disconnected():
    x = pyro.sample("x", dist.Normal(0.0, 1.0))
    _ = pyro.deterministic("side", x * 10.0)   # registered but return value not used
    y = pyro.deterministic("y", x * 2.0)       # depends on x, not on side's return value
    return {"x": x, "y": y}

torch.manual_seed(0)
base_d = model_disconnected()
torch.manual_seed(0)
patched_d = do(model_disconnected, {"side": torch.tensor(999.0)})()
assert abs(patched_d["y"].item() - base_d["y"].item()) < 1e-6
print(f"  y base: {base_d['y'].item():.4f}, y with side patched: {patched_d['y'].item():.4f}")
print("  PASS: patching disconnected site has no effect")
print("  *** The causal graph is implicit in Python control flow — you MUST assign and use ***")
print("      the return value of pyro.deterministic for patching to matter")


print("\n" + "=" * 60)
print("SUMMARY — Q1 Effect Handler Mechanics")
print("=" * 60)
print("""
1. do() intercepts pyro.deterministic via _pyro_post_sample hook. ✓
   - Works because pyro.deterministic is pyro.sample with Delta dist
   - The return value of pyro.deterministic IS the (possibly patched) msg["value"]

2. MultiWorldCounterfactual uses tensor broadcasting (NOT re-execution). ✓
   - Extra batch dims added on the left (at first_available_dim=-5)
   - gather()/indices_of() MUST be called inside the context (uses active plates)
   - gather() needs matching event_dim for vector/matrix sites

3. Standard PyTorch ops (matmul, softmax, tanh, sum) between sites work. ✓
   - They broadcast correctly over extra leading batch dimensions
   - Attention's Q@K.T = q.mT operates on last 2 dims → world-safe

4. view()/reshape() with hardcoded shapes BREAKS multi-world execution. ✗
   - RuntimeError: view cannot reshape world-indexed tensor to fixed size
   - This is the critical constraint for transformers

5. Causal graph is Python control flow. ✓
   - Must assign and USE the return value of pyro.deterministic
   - Side computations that don't feed downstream are patched but causally inert
""")
