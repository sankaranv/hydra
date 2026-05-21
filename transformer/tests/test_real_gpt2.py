"""Integration tests for HookedGPT2 with real GPT-2 large weights.

Loads the gpt2-large snapshot from disk (local_files_only=True) and verifies
shape correctness, site count, and do() / hook equivalence.

All tests are marked slow and excluded from fast CI runs.
Run with: pytest transformer/tests/test_real_gpt2.py -v -m slow
"""

import pytest
import torch

from transformer.lib.cache import run_and_cache
from transformer.lib.model import HookedGPT2, check_do_hook_equivalence, site_names

GPT2_LARGE_PATH = (
    "/datasets/ai/gpt/hub/models--openai-community--gpt2-large"
    "/snapshots/32b71b12589c2f8d625668d2335a01cac3249519"
)

# GPT-2 large architecture constants
N_LAYERS = 36
N_HEADS = 20
D_MODEL = 1280
VOCAB_SIZE = 50257
SEQ_LEN = 8
BATCH_SIZE = 1

# Total sites: resid_pre_0 + per-layer (n_heads + attn_out + mlp_out + resid_post)
EXPECTED_SITE_COUNT = 1 + N_LAYERS * (N_HEADS + 3)


# ---------------------------------------------------------------------------
# Module-scope fixture: load real GPT-2 large weights once for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_gpt2():
    model = HookedGPT2(GPT2_LARGE_PATH)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_gpt2_loads_and_forward_pass(real_gpt2):
    """Forward pass with a random token sequence produces the expected output shape."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    with torch.no_grad():
        output = real_gpt2(input_ids)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


@pytest.mark.slow
def test_real_gpt2_site_count(real_gpt2):
    """run_and_cache captures exactly the expected number of deterministic sites."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    with torch.no_grad():
        cache = run_and_cache(real_gpt2, input_ids)

    assert len(cache) == EXPECTED_SITE_COUNT

    # Cross-check against the site_names utility for an exact name-set match
    expected_names = set(site_names(N_LAYERS, N_HEADS))
    assert set(cache.keys()) == expected_names


@pytest.mark.slow
def test_real_gpt2_head_shape(real_gpt2):
    """head_0_0 has shape (batch, seq, d_model) — the full residual-stream contribution."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    with torch.no_grad():
        cache = run_and_cache(real_gpt2, input_ids, sites=["head_0_0"])

    assert cache["head_0_0"].shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


@pytest.mark.slow
def test_real_gpt2_do_hook_equivalence():
    """do() patching and native hook injection agree on head_0_0 zeroing."""
    result = check_do_hook_equivalence(GPT2_LARGE_PATH)
    assert result["passed"], (
        f"do() and hook disagree on gpt2-large: max_diff={result['max_diff']:.2e}"
    )
