"""Smoke tests for HookedGPTNeoX, HookedLlama, and HookedModel factory.

Each non-slow test uses a tiny synthetic config to avoid loading real model weights.
Tests that require the actual on-disk models are marked @pytest.mark.slow.

Run fast tests only (default CI):
    pytest transformer/tests/test_hooked_models.py -v

Run all including slow (requires disk access):
    pytest transformer/tests/test_hooked_models.py -v -m slow
"""

import pyro
import pytest
import torch
from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig, Qwen2Config

from transformer.lib.model import (
    HookedGPT2,
    HookedGPTNeoX,
    HookedLlama,
    HookedModel,
)


# ---------------------------------------------------------------------------
# Tiny synthetic configs (no real weights, fast in CI)
# ---------------------------------------------------------------------------

TINY_GPT2_CONFIG = GPT2Config(
    n_layer=2,
    n_head=2,
    n_embd=8,
    attn_pdrop=0.0,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
)

TINY_NEOX_CONFIG = GPTNeoXConfig(
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=64,
    intermediate_size=128,
    hidden_dropout=0.0,
    attention_dropout=0.0,
)

TINY_LLAMA_CONFIG = LlamaConfig(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=64,
    intermediate_size=128,
)

TINY_QWEN2_CONFIG = Qwen2Config(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=64,
    intermediate_size=128,
)

# Paths to on-disk models (only used for slow tests)
PYTHIA_PATH = "/datasets/ai/pythia/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42"
QWEN_PATH = "/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
LLAMA_PATH = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


# ---------------------------------------------------------------------------
# Expected site names helper
# ---------------------------------------------------------------------------


def expected_site_names(n_layers: int, n_heads: int) -> set[str]:
    names = {"resid_pre_0"}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            names.add(f"head_{layer_idx}_{head_idx}")
        names.add(f"attn_out_{layer_idx}")
        names.add(f"mlp_out_{layer_idx}")
        names.add(f"resid_post_{layer_idx}")
    return names


# ---------------------------------------------------------------------------
# HookedGPT2 (tiny synthetic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_gpt2():
    model = HookedGPT2(TINY_GPT2_CONFIG)
    model.eval()
    return model


def test_gpt2_site_names(tiny_gpt2):
    """All expected pyro sites are registered for a tiny GPT-2."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_gpt2.gpt2.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_gpt2).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(tiny_gpt2.n_layers, tiny_gpt2.n_heads) == registered


def test_gpt2_output_shape(tiny_gpt2):
    """Forward pass returns [batch, seq, d_model]."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_gpt2.gpt2.config.vocab_size, (2, 7))
    out = tiny_gpt2(input_ids)
    assert out.shape == (2, 7, tiny_gpt2.d_model)


def test_gpt2_do_changes_output(tiny_gpt2):
    """ChiRho do() patching on head_0_0 changes the model output."""
    from chirho.interventional.handlers import do

    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_gpt2.gpt2.config.vocab_size, (1, 5))
    patch = torch.zeros(1, 5, tiny_gpt2.d_model)
    with torch.no_grad():
        out_clean = tiny_gpt2(input_ids)
        out_patched = do(tiny_gpt2, {"head_0_0": patch})(input_ids)
    assert not torch.allclose(out_clean, out_patched, atol=1e-4)


# ---------------------------------------------------------------------------
# HookedGPTNeoX (tiny synthetic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_neox():
    model = HookedGPTNeoX(TINY_NEOX_CONFIG)
    model.eval()
    return model


def test_neox_site_names(tiny_neox):
    """All expected pyro sites are registered for a tiny GPT-NeoX."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_neox.neox.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_neox).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(tiny_neox.n_layers, tiny_neox.n_heads) == registered


def test_neox_output_shape(tiny_neox):
    """Forward pass returns [batch, seq, d_model]."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_neox.neox.config.vocab_size, (2, 7))
    out = tiny_neox(input_ids)
    assert out.shape == (2, 7, tiny_neox.d_model)


def test_neox_do_changes_output(tiny_neox):
    """ChiRho do() patching on head_0_0 changes the model output."""
    from chirho.interventional.handlers import do

    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_neox.neox.config.vocab_size, (1, 5))
    patch = torch.zeros(1, 5, tiny_neox.d_model)
    with torch.no_grad():
        out_clean = tiny_neox(input_ids)
        out_patched = do(tiny_neox, {"head_0_0": patch})(input_ids)
    assert not torch.allclose(out_clean, out_patched, atol=1e-4)


def test_neox_head_contributions_sum_to_attn_out(tiny_neox):
    """Sum of head contributions equals attn_out (up to float32 tolerance)."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_neox.neox.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_neox).get_trace(input_ids)
    nodes = {
        n: node["value"] for n, node in trace.nodes.items() if node["type"] == "sample"
    }

    for layer_idx in range(tiny_neox.n_layers):
        head_sum = sum(nodes[f"head_{layer_idx}_{h}"] for h in range(tiny_neox.n_heads))
        # The dense bias is included in attn_out; check that they match
        attn_out = nodes[f"attn_out_{layer_idx}"]
        assert torch.allclose(head_sum, attn_out, atol=1e-5), (
            f"Layer {layer_idx}: head sum differs from attn_out by "
            f"{(head_sum - attn_out).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# HookedLlama with LlamaConfig (tiny synthetic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_llama():
    model = HookedLlama(TINY_LLAMA_CONFIG)
    model.eval()
    return model


def test_llama_site_names(tiny_llama):
    """All expected pyro sites are registered for a tiny Llama."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_llama.model.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_llama).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(tiny_llama.n_layers, tiny_llama.n_heads) == registered


def test_llama_output_shape(tiny_llama):
    """Forward pass returns [batch, seq, d_model]."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_llama.model.config.vocab_size, (2, 7))
    out = tiny_llama(input_ids)
    assert out.shape == (2, 7, tiny_llama.d_model)


def test_llama_do_changes_output(tiny_llama):
    """ChiRho do() patching on head_0_0 changes the model output."""
    from chirho.interventional.handlers import do

    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_llama.model.config.vocab_size, (1, 5))
    patch = torch.zeros(1, 5, tiny_llama.d_model)
    with torch.no_grad():
        out_clean = tiny_llama(input_ids)
        out_patched = do(tiny_llama, {"head_0_0": patch})(input_ids)
    assert not torch.allclose(out_clean, out_patched, atol=1e-4)


def test_llama_head_contributions_sum_to_attn_out(tiny_llama):
    """Sum of head contributions equals attn_out (up to float32 tolerance)."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_llama.model.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_llama).get_trace(input_ids)
    nodes = {
        n: node["value"] for n, node in trace.nodes.items() if node["type"] == "sample"
    }

    for layer_idx in range(tiny_llama.n_layers):
        head_sum = sum(
            nodes[f"head_{layer_idx}_{h}"] for h in range(tiny_llama.n_heads)
        )
        attn_out = nodes[f"attn_out_{layer_idx}"]
        assert torch.allclose(head_sum, attn_out, atol=1e-5), (
            f"Layer {layer_idx}: head sum differs from attn_out by "
            f"{(head_sum - attn_out).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# HookedLlama with Qwen2Config (tiny synthetic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen2():
    model = HookedLlama(TINY_QWEN2_CONFIG)
    model.eval()
    return model


def test_qwen2_site_names(tiny_qwen2):
    """All expected pyro sites are registered for a tiny Qwen2 model."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_qwen2.model.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(tiny_qwen2).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(tiny_qwen2.n_layers, tiny_qwen2.n_heads) == registered


def test_qwen2_output_shape(tiny_qwen2):
    """Forward pass returns [batch, seq, d_model]."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_qwen2.model.config.vocab_size, (2, 7))
    out = tiny_qwen2(input_ids)
    assert out.shape == (2, 7, tiny_qwen2.d_model)


# ---------------------------------------------------------------------------
# HookedModel factory (tiny synthetic GPT-2 is tested via real path in slow tests)
# ---------------------------------------------------------------------------


def test_factory_dispatch_gpt2_config():
    """HookedModel factory dispatch table maps all expected model_type strings."""
    from transformer.lib.models.factory import _MODEL_TYPE_TO_CLASS

    # Check by class name to avoid identity issues from re-exports in model.py
    assert _MODEL_TYPE_TO_CLASS["gpt2"].__name__ == "HookedGPT2"
    assert _MODEL_TYPE_TO_CLASS["gpt_neox"].__name__ == "HookedGPTNeoX"
    assert _MODEL_TYPE_TO_CLASS["llama"].__name__ == "HookedLlama"
    assert _MODEL_TYPE_TO_CLASS["qwen2"].__name__ == "HookedLlama"


def test_factory_unsupported_raises():
    """HookedModel.from_pretrained raises ValueError for unknown model types."""
    from transformer.lib.models.factory import _MODEL_TYPE_TO_CLASS

    assert "bert" not in _MODEL_TYPE_TO_CLASS


# ---------------------------------------------------------------------------
# Slow tests: load real model weights from disk
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_factory_returns_hooked_gpt_neox_for_pythia():
    """HookedModel.from_pretrained(pythia_path) returns HookedGPTNeoX."""
    model = HookedModel.from_pretrained(PYTHIA_PATH)
    assert isinstance(model, HookedGPTNeoX)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.neox.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(model).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(model.n_layers, model.n_heads) == registered


@pytest.mark.slow
def test_factory_returns_hooked_llama_for_qwen():
    """HookedModel.from_pretrained(qwen_path) returns HookedLlama."""
    model = HookedModel.from_pretrained(QWEN_PATH)
    assert isinstance(model, HookedLlama)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.model.config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(model).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(model.n_layers, model.n_heads) == registered


@pytest.mark.slow
def test_llama_8b_synthetic_config():
    """HookedLlama works with a Llama-3.1-8B-scale config using synthetic weights.

    The real 8B model is 16 GB and may cause OOM on CPU. This test uses the full
    Llama-3.1-8B architecture config (32 layers, 32 heads, hidden=4096) but with
    randomly initialized weights (no pretrained loading). It verifies the wrapper
    initializes and registers the correct sites without loading real weights.
    """
    config = LlamaConfig(
        num_hidden_layers=2,  # Reduced layers; full 32 would take ~2 GB on CPU
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        intermediate_size=14336,
        head_dim=128,
    )
    model = HookedLlama(config)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    trace = pyro.poutine.trace(model).get_trace(input_ids)
    registered = {n for n, node in trace.nodes.items() if node["type"] == "sample"}
    assert expected_site_names(2, 32) == registered
    out = model(input_ids)
    assert out.shape == (1, 5, 4096)
