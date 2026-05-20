# ChiRho × Transformer Integration: Technical Feasibility Report

**Date:** 2026-05-20  
**Environment:** chirho 0.3.0, pyro 1.9.1, torch 2.12.0, nnsight 0.7.0, transformers 5.9.0  
**No TransformerLens available** — investigation uses HuggingFace transformers + nnsight.  
**Probe scripts:** `transformer/q1_handler_mechanics.py` through `q5_nnsight_compatibility.py`

---

## 1. Effect Handler Contract

### What `do` and `MultiWorldCounterfactual` intercept

`do` (the `Interventions` class) hooks into `_pyro_post_sample`, which fires after **any** `pyro.sample` call — including `pyro.deterministic`, which is syntactic sugar for `pyro.sample` with a Delta distribution. The return value of `pyro.deterministic("name", value)` IS the (possibly replaced) `msg["value"]` after handlers have run. This is what makes structural intervention at deterministic sites work.

**The contract a Pyro program must satisfy for ChiRho to work:**

1. **Register with `pyro.deterministic(name, value, event_dim=N)`** at the desired intervention points.
2. **Assign and use the return value** — not the pre-registered value. The causal graph is implicit in Python control flow, not in Pyro naming. A side computation whose return value is discarded is patched at the trace level but causally inert.
3. **PyTorch ops between sites must support broadcasting over extra batch dims.** `MultiWorldCounterfactual` adds extra dimensions on the left (at `first_available_dim`, default −5). Standard ops — `nn.Linear`, `@` (matmul), `softmax`, `tanh`, `LayerNorm` — are world-safe because they operate on the last (event) dimensions. `view(fixed_size)` and `reshape(fixed_size)` are **not** world-safe and will raise `RuntimeError`.
4. **`gather()` and `indices_of()` must be called inside the `MultiWorldCounterfactual` context.** They consult the active handler stack for plate metadata. Outside the context they are no-ops.
5. **`event_dim` must be passed consistently.** `pyro.deterministic("h", hidden, event_dim=2)` declares a [seq, d_head] event. Subsequent `gather(result["h"], ..., event_dim=2)` must match. For scalar outputs, `event_dim=0`. Mismatched event_dim causes gather to silently fail or index the wrong dimension.
6. **`first_available_dim` must be negative enough** to accommodate event dims. For sites with `event_dim=N`, the world dimension needs at least `first_available_dim ≤ -(5 + N)`. Default `−5` works only for scalar sites; per-head activations with event_dim=2 need at least `−7`.

**Critical gotcha:** `MultiWorldCounterfactual` uses **tensor broadcasting**, not re-execution. Each `intervene` call adds one extra batch dimension. Memory scales as O(K^N) where N = number of intervention sites and K = number of worlds per site. `SearchForExplanation` introduces 3 worlds per antecedent.

---

## 2. Proof-of-Concept Result

### Does it work?

**Yes, completely.** A 2-layer attention-only transformer wrapped as a Pyro model passes all correctness checks.

**Key correctness test passed (Probe Q2):**  
Patching `head_0_0` to zeros via `do()` gives identical output to injecting zeros via a manual PyTorch forward hook:
```
Max absolute difference: 0.00e+00
```

**`MultiWorldCounterfactual` gives correct factual and intervened worlds simultaneously:**
```
out_factual norm:     7.3495   (matches clean run: True)
out_intervened norm:  6.8325   (matches hook-patched run: True)
```

### How the wrapper is structured

The wrapper avoids all shape-assumption-breaking operations:

```python
class AttentionHead(nn.Module):
    def forward(self, x):
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
        # q.mT transposes last 2 dims — world-safe regardless of leading batch dims
        attn = torch.softmax(q @ k.mT * self.scale, dim=-1)
        return attn @ v

class AttentionLayer(nn.Module):
    def forward(self, x, layer_idx):
        resid = x
        for h_idx, head in enumerate(self.heads):
            head_out = head(x)
            # Register and USE the return value
            head_out = pyro.deterministic(f"head_{layer_idx}_{h_idx}", head_out, event_dim=2)
            resid = resid + self.W_O(head_out)
        return resid
```

**Sites registered in the Pyro trace:**
- `head_L_H`: shape `[batch, seq, d_head]` — per-head output (OV circuit)
- `resid_L`: shape `[batch, seq, d_model]` — residual stream per layer

### The GPT-2 reshape problem

The proof-of-concept works because the attention is written without `view/reshape`. **Real GPT-2 uses `view` to split heads:**

```python
# GPT-2 actual attention (simplified)
q = q.view(batch, seq, n_heads, d_head).transpose(1, 2)  # view() breaks multi-world
```

This will raise `RuntimeError` when world-indexed tensors have extra leading dimensions. Solutions (in order of invasiveness):

1. **Rewrite GPT-2 attention** to use `einsum` or `reshape(-1, ...)` — avoids hardcoded shapes. Estimated effort: 2–4 hours per model.
2. **Intercept after the reshape, not before.** Register `pyro.deterministic` sites at points where the tensor has already been reshaped to its final per-head shape, and keep the ChiRho intervention downstream of all shape-manipulating ops.
3. **Use nnsight's hook injection** to bypass the reshape problem entirely — hook at the head output after all shape manipulation is done, register the result as a `pyro.deterministic` site, and propagate from there. This works because nnsight hooks fire after the module's forward completes (shapes are already resolved).

---

## 3. SearchForExplanation Feasibility

### Can it be called on transformer-scale graphs with continuous activations?

**Mechanically yes; scientifically problematic without modifications.**

**What works:**
- `SearchForExplanation` accepts continuous antecedents. For `dist.Normal` sites, `ExtractSupports` returns `constraints.real`; for vector sites, `constraints.independent(real, 1)`.
- The case variable (0=factual, 1=necessity, 2=sufficiency) IS introduced for continuous antecedents.
- Vector-valued antecedents (d_head=8) work; PNS can be estimated from case variable posterior.

**Immediate failure modes:**

**1. The random alternative proposal uses Cauchy distribution for real-valued support.**

```python
uniform_proposal(constraints.real, event_shape=()) 
# → MaskedDistribution wrapping Cauchy
# Sample: -26.03 (observed in Q4 probe)
```

`uniform_proposal` for `constraints.real` returns a Cauchy distribution (heavy tails, undefined variance). For transformer activations that are bounded by LayerNorm (typically in [-10, 10]), this produces pathological "necessity test" alternatives far outside the normal activation range. The necessity check — "would Y differ if X had been *something else*?" — becomes trivially true because the alternative is extreme.

**Fix required:** Override `alternatives` with a domain-appropriate proposal, e.g., mean-ablation (replace with dataset mean) or Gaussian centered on the activation distribution.

**2. Case=2 (sufficiency world) never fires without proper inference.**

In Q4 Probe 3 with vector antecedents:
```
Case distribution: 0=110 (factual), 1=90 (necessity), 2=0 (sufficiency)
```

Without an inference guide, the case variable is sampled uniformly from Categorical(3), and the sufficiency intervention (hold antecedent to its observed value in the counterfactual world) fails to receive probability mass. Meaningful PNS estimation requires SVI with a guide that conditions on the consequent evidence.

**3. Memory explosion with multiple antecedents.**

Each antecedent adds 3 worlds (factual, necessity, sufficiency). Memory scales as O(3^N) × activation tensor size:

| N antecedents | Worlds | GPT-2 small cost |
|---|---|---|
| 1 | 3 | 216 MB |
| 2 | 9 | 648 MB |
| 3 | 27 | 1.9 GB |
| 5 | 243 | 17.5 GB |

At GPT-2 scale (12 layers × 12 heads = 144 candidate antecedents), joint inference is computationally infeasible. `SearchForExplanation` must be applied to **subsets of 1–2 candidate heads at a time**, not jointly.

**4. The `undo_split` function is exponential in `len(antecedents)`.**

In `components.py`, `undo_split` has a comment: "TODO exponential in len(antecedents)". The witness preemption logic constructs a list of all world combinations, which is O(3^N). This is the hard algorithmic bound.

**5. `consequent_scale` calibration.**

For continuous outputs (logits, probability differences), the default `consequent_scale=1e-2` is very soft. In Q4 experiments, PNS estimates were insensitive to scale (0.42–0.52 across 4 orders of magnitude) because without inference the consequent factors don't constrain the posterior. With proper SVI, scale must be calibrated to the output variance (e.g., set to `1 / std(logit difference)` across the dataset).

---

## 4. nnsight Compatibility

**Result: fully compatible with local dispatch.**

**Key finding from Q5 Probe 1:** nnsight 0.7 with local dispatch executes the forward pass synchronously inside the `trace()` context. Pyro stack depth during nnsight execution = 1 (with handler active). This means:

- Pyro handlers (including ChiRho's `do`, `Interventions`, `MultiWorldCounterfactual`) ARE on the stack when the model's `forward()` runs inside `nnsight.trace()`.
- `pyro.deterministic` sites registered inside `model.forward()` appear in `pyro.poutine.trace` output when nested inside `nnsight.trace()`.
- `do()` patching and `MultiWorldCounterfactual` compose correctly with nnsight's trace context (verified in Q5 Probes 3–4).

**What breaks:** Remote/deferred dispatch (nnsight connecting to a remote model server) would defer execution until after the `trace()` context exits, at which point Pyro handlers are no longer active. Always use local dispatch when composing with ChiRho.

**Two viable architectures:**

**Option A — Tight coupling (sites inside model.forward):**
```python
# Register activation sites directly inside the transformer's forward()
head_out = pyro.deterministic("head_0_0", head_out, event_dim=2)

# Wrap with NNsight for convenience; ChiRho handlers compose transparently
nnsight_model = NNsight(transformer)
with MultiWorldCounterfactual():
    with do(actions={"head_0_0": patch_val}):
        with nnsight_model.trace(x):
            out = nnsight_model.output.save()
```

**Option B — Clean separation (recommended for production):**
```python
# Step 1: nnsight extracts activations
with nnsight_model.trace(x_corrupt):
    h_saved = nnsight_model.attn_layers[0].heads[0].output.save()
h_corrupt = h_saved.detach().clone()

# Step 2: ChiRho evaluates AC on extracted values
def ac_model(h_cached):
    h = pyro.deterministic("h", h_cached, event_dim=2)
    logit = pyro.deterministic("logit", lm_head(residual + W_O @ h), event_dim=0)
    return {"h": h, "logit": logit}

with MultiWorldCounterfactual():
    with SearchForExplanation(...):
        result = ac_model(h_corrupt)
```

Option B is more modular and avoids any risk of execution model coupling. Option A is simpler to implement but requires the model to be aware of Pyro.

---

## 5. Honest Cost Estimate

### To get a working system on a toy 2-layer model (done in this investigation)

**Already complete.** The proof-of-concept in `q2_transformer_wrapper.py` demonstrates:
- Shape-assumption-free attention-only transformer
- `do()` correctness verified against forward hooks
- `MultiWorldCounterfactual` gives correct world-indexed tensors
- `pyro.poutine.trace` works as the activation caching mechanism
- nnsight + ChiRho compose with local dispatch

**Remaining for toy model to run `SearchForExplanation` end-to-end:** ~2–3 days
1. Write a proper SVI guide for the AC model
2. Replace Cauchy proposal with domain-appropriate alternative
3. Calibrate `consequent_scale` to activation statistics
4. Verify PNS estimates are interpretable for a known causal structure

### To get GPT-2 working

**Engineering work (1–2 weeks):**
- Rewrite GPT-2 attention to avoid `view/reshape` with fixed shapes. This means replacing the standard HuggingFace `GPT2Attention` with a ChiRho-compatible implementation. The embedding, LM head, and positional encoding are all safe.
- Alternatively: hook at the per-head output after reshaping (Option B), which avoids rewriting the model but requires careful hook registration.
- Calibrate `first_available_dim` for the model's activation shapes.
- Test memory usage at realistic sequence lengths (512–1024 tokens).

**Research work (requires ChiRho modifications):**
- Implement a sequential antecedent search strategy (test heads one at a time, not jointly).
- Replace Cauchy proposal with an activation-informed proposal (e.g., Laplace or Gaussian matched to observed activation statistics).
- Implement `auto_consequent_scale` that calibrates to the output logit variance.

### What requires modifications to ChiRho itself

1. **`undo_split` is exponential in antecedents.** The TODO comment is in the source. Joint inference over N antecedents requires O(3^N) tensor operations. A sequential approximation (one antecedent at a time, with fixed witness values) would require a new handler or a wrapper pattern around `SearchForExplanation`.

2. **`random_intervention` for `constraints.real` uses Cauchy.** For bounded activations, this is a bad proposal. ChiRho would need a way to override the proposal per-site without rebuilding `SearchForExplanation` from scratch. Currently the `alternatives` parameter accepts a custom intervention, so this is workable without modifying ChiRho — you pass a custom callable.

3. **No built-in sequential search.** `SearchForExplanation` is designed for joint inference over a small set of antecedents. A sequential search that tests each antecedent independently and accumulates evidence is not built in but can be implemented on top of the existing API.

---

## 6. Verdict on the Approach

**The basic machinery composes.** ChiRho can be applied to a transformer forward pass. The effect handler system works exactly as documented, and the key correctness test (do() == hook injection) passes with zero numerical error.

**The hard problems are:**

1. **The reshape problem for production models.** GPT-2's `view()`-based attention reshaping breaks multi-world execution. This is solvable with a model rewrite or careful hook placement, but it's real engineering work — not a theoretical obstacle.

2. **Joint inference over many antecedents is infeasible at transformer scale.** Memory grows as O(3^N). `SearchForExplanation` must be applied sequentially (one head at a time), which changes the AC semantics: you're testing individual heads against fixed witnesses, not jointly searching for a minimal causing subset. Whether this is scientifically adequate depends on what "backup mechanism" means precisely.

3. **The Cauchy proposal for continuous activations.** The default `random_intervention` for real-valued sites draws from Cauchy, which produces extreme alternative values outside any realistic activation range. This makes necessity trivially true for almost any head. Custom alternatives must be supplied.

4. **Inference is required for meaningful PNS.** Running `SearchForExplanation` without a proper SVI guide gives case variable samples that don't concentrate on the actual cause. The guide must be structured to condition correctly on both antecedent and consequent evidence.

**Is there a better path?**

For a narrow backup mechanism detector targeting a specific task, **implementing the AC search logic directly on top of nnsight** (without Pyro) would be faster to build and easier to debug. The AC condition (necessity AND sufficiency) can be evaluated with two explicit forward passes:
- Necessity: run with head H patched to zero (or to mean) → does Y change?
- Sufficiency: run normal input but with head H output held to its factual value in a counterfactual where the input is corrupted → does Y survive?

This is the approach Mueller (2024) and pyvene take. It sacrifices the principled probabilistic framework (no joint inference, no posterior over causes) but is computationally feasible and conceptually transparent.

**Recommendation:** Use ChiRho if the research question requires principled probabilistic AC (joint posteriors over causal subsets, uncertainty quantification, integration with the Pyro inference ecosystem). Accept the cost of model rewriting and sequential search. Skip ChiRho and use nnsight + direct AC computation if you need results on production GPT-2 within a week.

---

## Appendix: Files Produced

| File | Content |
|---|---|
| `q1_handler_mechanics.py` | Effect handler internals: do(), MultiWorldCounterfactual, view/reshape failure |
| `q2_transformer_wrapper.py` | Minimal 2-layer transformer wrapper, correctness verification |
| `q3_activation_caching.py` | poutine.trace as caching layer, multi-world memory analysis |
| `q4_search_for_explanation.py` | Continuous antecedents, Cauchy proposal, scale test, scalability |
| `q5_nnsight_compatibility.py` | Eager dispatch verification, tight/loose coupling architectures |
