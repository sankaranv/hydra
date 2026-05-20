"""Integration tests for transformer/lib.

Uses a 2-layer 2-head attention-only toy transformer as a cheap fixture.
All tests verify the library API without requiring network access or GPU.

Run with: pytest transformer/tests/test_lib.py -v
"""

import pyro
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config

from transformer.lib.ac_query import run_ac_query
from transformer.lib.cache import run_and_cache
from transformer.lib.interventions import (
    build_alternatives,
    mean_ablation,
    resample_ablation,
    zero_ablation,
)
from transformer.lib.model import HookedGPT2, check_do_hook_equivalence, site_names
from transformer.lib.prefilter import logit_diff_attribution, rank_candidates
from transformer.lib.verdict import (
    compute_necessity,
    compute_pns,
    compute_sufficiency,
    extract_case_variables,
    summarize_verdict,
)


# ---------------------------------------------------------------------------
# Shared fixture: tiny attention-only transformer with pyro.deterministic sites
# ---------------------------------------------------------------------------


class _AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.scale = d_head**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
        return torch.softmax(q @ k.mT * self.scale, dim=-1) @ v


class _AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [_AttentionHead(d_model, d_head) for _ in range(n_heads)]
        )
        self.W_O = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        resid = x
        for h, head in enumerate(self.heads):
            h_out = head(x)
            h_out = pyro.deterministic(f"head_{layer_idx}_{h}", h_out, event_dim=2)
            resid = resid + self.W_O(h_out)
        return resid


class _TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 8, n_heads: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_AttentionLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        for layer_idx, layer in enumerate(self.layers):
            resid = layer(resid, layer_idx)
            resid = pyro.deterministic(f"resid_{layer_idx}", resid, event_dim=2)
        return resid


@pytest.fixture(scope="module")
def tiny_model() -> _TinyTransformer:
    torch.manual_seed(42)
    model = _TinyTransformer(d_model=8, n_heads=2, n_layers=2)
    model.eval()
    return model


@pytest.fixture(scope="module")
def x_in() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, 5, 8)


@pytest.fixture(scope="module")
def model_fn(tiny_model, x_in):
    """Zero-argument closure over tiny_model and x_in, with logit site."""

    def _fn():
        inp = pyro.deterministic("input", x_in, event_dim=2)
        out = tiny_model(inp)
        return pyro.deterministic("logits", out, event_dim=2)

    return _fn


# ---------------------------------------------------------------------------
# cache.py tests
# ---------------------------------------------------------------------------


def test_run_and_cache_returns_all_sites(tiny_model, x_in):
    def model_fn(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        return tiny_model(inp)

    cache = run_and_cache(model_fn, x_in)
    assert "head_0_0" in cache
    assert "resid_1" in cache


def test_run_and_cache_site_filter(tiny_model, x_in):
    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    filtered = run_and_cache(model_fn, x_in, sites=["head_0_0", "resid_0"])
    assert set(filtered.keys()) == {"head_0_0", "resid_0"}


def test_run_and_cache_deterministic(tiny_model, x_in):
    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    cache_a = run_and_cache(model_fn, x_in)
    cache_b = run_and_cache(model_fn, x_in)
    assert torch.allclose(cache_a["head_0_0"], cache_b["head_0_0"])


# ---------------------------------------------------------------------------
# interventions.py tests
# ---------------------------------------------------------------------------


def test_zero_ablation_shape(tiny_model, x_in):
    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    cache = run_and_cache(model_fn, x_in, sites=["head_0_0", "head_0_1"])
    zeros = zero_ablation(cache)
    assert all(v.sum() == 0.0 for v in zeros.values())
    assert zeros["head_0_0"].shape == cache["head_0_0"].shape


def test_mean_ablation_shape(tiny_model):
    x1, x2, x3 = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8)

    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    # Head sites in the toy transformer are [batch, seq, d_head] where d_head = d_model // n_heads
    reference = run_and_cache(model_fn, x1, sites=["head_0_0"])
    expected_shape = reference["head_0_0"].shape

    means = mean_ablation(model_fn, inputs=[(x1,), (x2,), (x3,)], sites=["head_0_0"])
    assert means["head_0_0"].shape == expected_shape


def test_resample_ablation_shape(tiny_model):
    x_base, x_ref0, x_ref1 = (
        torch.randn(1, 5, 8),
        torch.randn(1, 5, 8),
        torch.randn(1, 5, 8),
    )

    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    base_cache = run_and_cache(model_fn, x_base, sites=["head_0_0"])
    ref0 = run_and_cache(model_fn, x_ref0, sites=["head_0_0"])
    ref1 = run_and_cache(model_fn, x_ref1, sites=["head_0_0"])
    stacked_ref = {"head_0_0": torch.stack([ref0["head_0_0"], ref1["head_0_0"]])}
    resampled = resample_ablation(base_cache, stacked_ref)
    assert resampled["head_0_0"].shape == base_cache["head_0_0"].shape


def test_build_alternatives_zero_dispatch(tiny_model, x_in):
    def model_fn(x):
        return tiny_model(pyro.deterministic("input", x, event_dim=2))

    cache = run_and_cache(model_fn, x_in, sites=["head_0_0"])
    alts = build_alternatives(["head_0_0"], method="zero", cache=cache)
    assert alts["head_0_0"].sum() == 0.0


# ---------------------------------------------------------------------------
# ac_query.py tests
# ---------------------------------------------------------------------------


def test_run_ac_query_returns_traces_with_case_variables(model_fn, x_in, tiny_model):
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alt = zero_ablation({"head_0_0": cache["head_0_0"]})

    traces = run_ac_query(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alt,
        witnesses=None,
        consequent_site="logits",
        consequent_value=cache["logits"],
        consequent_scale=0.1,
        num_samples=20,
    )

    assert len(traces) == 20
    case_key = "__cause____antecedent_head_0_0"
    case_vals = [tr[case_key].item() for tr in traces if case_key in tr]
    assert len(case_vals) == 20
    assert all(v in {0, 1, 2} for v in case_vals)


def test_run_ac_query_accepts_model_args(tiny_model, x_in):
    """run_ac_query *model_args variant: model_fn takes x as argument."""

    def model_fn_with_args(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        out = tiny_model(inp)
        return pyro.deterministic("logits", out, event_dim=2)

    cache = run_and_cache(model_fn_with_args, x_in, sites=["head_0_0", "logits"])
    alt = zero_ablation({"head_0_0": cache["head_0_0"]})

    traces = run_ac_query(
        model_fn_with_args,
        x_in,  # passed as *model_args
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alt,
        witnesses=None,
        consequent_site="logits",
        consequent_value=cache["logits"],
        consequent_scale=0.1,
        num_samples=10,
    )

    assert len(traces) == 10
    case_key = "__cause____antecedent_head_0_0"
    assert all(case_key in tr for tr in traces)


# ---------------------------------------------------------------------------
# verdict.py tests
# ---------------------------------------------------------------------------


def test_extract_case_variables_shape(model_fn, x_in):
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alt = zero_ablation({"head_0_0": cache["head_0_0"]})
    traces = run_ac_query(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alt,
        witnesses=None,
        consequent_site="logits",
        consequent_value=cache["logits"],
        consequent_scale=0.1,
        num_samples=30,
    )
    case_vars = extract_case_variables(traces)
    assert "head_0_0" in case_vars
    assert case_vars["head_0_0"].shape == torch.Size([30])


def test_pns_equals_necessity_plus_sufficiency(model_fn, x_in):
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alt = zero_ablation({"head_0_0": cache["head_0_0"]})
    traces = run_ac_query(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alt,
        witnesses=None,
        consequent_site="logits",
        consequent_value=cache["logits"],
        consequent_scale=0.1,
        num_samples=50,
    )
    case_vars = extract_case_variables(traces)
    pns = compute_pns(case_vars["head_0_0"])
    nec = compute_necessity(case_vars["head_0_0"])
    suf = compute_sufficiency(case_vars["head_0_0"])
    assert 0.0 <= pns <= 1.0
    assert abs(pns - (nec + suf)) < 1e-6


def test_summarize_verdict_absent_site(model_fn, x_in):
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alt = zero_ablation({"head_0_0": cache["head_0_0"]})
    traces = run_ac_query(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alt,
        witnesses=None,
        consequent_site="logits",
        consequent_value=cache["logits"],
        consequent_scale=0.1,
        num_samples=30,
    )
    verdict = summarize_verdict(traces, antecedent_sites=["head_0_0", "head_0_1"])
    assert verdict["head_0_0"]["n_samples"] == 30
    assert verdict["head_0_1"]["n_samples"] == 0


# ---------------------------------------------------------------------------
# prefilter.py tests
# ---------------------------------------------------------------------------


def test_rank_candidates_default_norm(model_fn, x_in):
    all_sites = ["head_0_0", "head_0_1", "head_1_0", "head_1_1"]
    cache = run_and_cache(model_fn, sites=all_sites)
    ranked = rank_candidates(cache, top_k=3)
    assert len(ranked) == 3
    assert all(s in all_sites for s in ranked)


def test_rank_candidates_custom_fn(model_fn, x_in):
    all_sites = ["head_0_0", "head_0_1", "head_1_0", "head_1_1"]
    cache = run_and_cache(model_fn, sites=all_sites)

    def inverse_norm(c):
        return {name: -act.norm().item() for name, act in c.items()}

    norm_ranked = rank_candidates(cache, top_k=4)
    inv_ranked = rank_candidates(cache, ranking_fn=inverse_norm, top_k=4)
    assert norm_ranked != inv_ranked


def test_logit_diff_attribution_non_negative(tiny_model, x_in):
    def model_fn(x):
        inp = pyro.deterministic("input", x, event_dim=2)
        out = tiny_model(inp)
        return pyro.deterministic("logits", out, event_dim=2)

    sites = ["head_0_0", "head_0_1", "head_1_0", "head_1_1"]
    attributions = logit_diff_attribution(
        model_fn=model_fn,
        input_args=(x_in,),
        sites=sites,
        correct_token_id=0,
        incorrect_token_id=1,
        logit_site="logits",
    )
    assert set(attributions.keys()) == set(sites)
    assert all(v >= 0.0 for v in attributions.values())


# ---------------------------------------------------------------------------
# model.py tests (HookedGPT2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_hooked_model():
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=8,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = HookedGPT2(config)
    model.eval()
    return model


def test_hooked_gpt2_all_sites_in_trace(tiny_hooked_model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_hooked_model.gpt2.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_hooked_model).get_trace(input_ids)
    registered = {
        name for name, node in trace.nodes.items() if node["type"] == "sample"
    }
    expected = set(site_names(tiny_hooked_model.n_layers, tiny_hooked_model.n_heads))
    assert expected == registered


def test_hooked_gpt2_do_changes_output(tiny_hooked_model):
    from chirho.interventional.handlers import do

    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_hooked_model.gpt2.config.vocab_size, (1, 5))
    patch = torch.zeros(1, 5, tiny_hooked_model.d_model)

    with torch.no_grad():
        out_clean = tiny_hooked_model(input_ids)
        out_patched = do(tiny_hooked_model, {"head_0_0": patch})(input_ids)

    assert not torch.allclose(out_clean, out_patched, atol=1e-4)


def test_check_do_hook_equivalence_tiny():
    result = check_do_hook_equivalence("tiny")
    assert result["passed"], (
        f"max_diff={result['max_diff']:.2e} — do() and hook disagree"
    )
    assert result["max_diff"] == 0.0
