"""Phase 4: end-to-end AC pipeline with SVI guide on a planted causal structure.

Ground truth model: logits = W @ head_0_0, head_0_1 is always zero and disconnected
from the output. Under this structure:
  - head_0_0 is both necessary (ablating it changes logits) and sufficient (it alone
    produces logits when witnesses are zeroed).
  - head_0_1 is neither necessary nor sufficient (ablating or keeping it leaves logits
    unchanged).

The test verifies that train_ac_guide + read_guide_verdict correctly identifies
head_0_0 as a high-PNS, high-sufficiency cause and head_0_1 as a low-PNS non-cause.
This validates that the sufficiency world fires reliably — unlike raw run_ac_query
which samples case values from the prior (uniformly) and gives ~2/3 PNS everywhere.

Run with: pytest transformer/tests/test_phase4.py -v
"""

import pyro
import pytest
import torch
import torch.nn as nn

from transformer.lib.ac_query import train_ac_guide
from transformer.lib.cache import run_and_cache
from transformer.lib.interventions import zero_ablation
from transformer.lib.verdict import read_guide_verdict


# ---------------------------------------------------------------------------
# Planted causal model
# ---------------------------------------------------------------------------


class _PlantedModel(nn.Module):
    """Two registered sites; only head_0_0 drives the output.

    logits = W @ head_0_0  (linear, so zero-ablating head_0_0 drives logits to 0)
    head_0_1 is always the zero tensor — ablating it leaves logits unchanged.
    """

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head_0_0: causal — directly derived from input
        head_0_0 = pyro.deterministic("head_0_0", x, event_dim=2)
        # head_0_1: non-causal — always zero, unconnected to output
        head_0_1 = pyro.deterministic("head_0_1", torch.zeros_like(x), event_dim=2)
        # logits depend only on head_0_0; head_0_1 is silenced
        _ = head_0_1
        return pyro.deterministic("logits", self.W(head_0_0), event_dim=2)


@pytest.fixture(scope="module")
def planted() -> _PlantedModel:
    return _PlantedModel(d=8).eval()


@pytest.fixture(scope="module")
def x_planted() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, 5, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_fn(planted, x):
    def model_fn():
        return planted(x)

    return model_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_guide_causal_head_has_high_pns(planted, x_planted):
    """head_0_0 must have PNS > 0.5 after guide training.

    PNS = P(case=0 | evidence) where case=0 means the antecedent intervention is
    NOT preempted (the causal world is active). For the causal head, both necessity
    and sufficiency worlds explain the evidence well, so the guide concentrates
    mass on case=0 for head_0_0.
    """
    model_fn = _make_model_fn(planted, x_planted)
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alts = zero_ablation({"head_0_0": cache["head_0_0"]})

    train_ac_guide(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alts,
        # witnesses={} prevents auto-discovery which would make head_0_0 both antecedent
        # and witness (since supports only covers head_0_0 and logits), creating NaN
        # gradients from the conflicting case variables. With no witnesses, the
        # sufficiency world keeps head_0_0 at its observed value and leaves everything
        # else at factual — still a valid and meaningful AC query.
        witnesses={},
        consequent_site="logits",
        consequent_value=cache["logits"],
        event_dim=2,
        # consequent_scale=1.0 gives sharper conditioning than the default 0.1;
        # the planted model's logit change is large, so a tighter scale is needed
        # to cleanly separate necessity (logits→0) from factual (logits unchanged).
        consequent_scale=1.0,
        n_steps=300,
        lr=0.05,
    )
    verdict = read_guide_verdict(["head_0_0"])

    assert verdict["head_0_0"]["pns"] > 0.5, (
        f"Expected PNS > 0.5 for causal head, got {verdict['head_0_0']['pns']:.3f}"
    )


def test_guide_noncausal_head_has_low_pns(planted, x_planted):
    """head_0_1 is always zero; ablating it leaves logits unchanged → low PNS."""
    model_fn = _make_model_fn(planted, x_planted)
    cache = run_and_cache(model_fn, sites=["head_0_1", "logits"])
    alts = zero_ablation({"head_0_1": cache["head_0_1"]})

    train_ac_guide(
        model_fn,
        antecedents={"head_0_1": cache["head_0_1"]},
        alternatives=alts,
        witnesses={},
        consequent_site="logits",
        consequent_value=cache["logits"],
        event_dim=2,
        consequent_scale=1.0,
        n_steps=300,
        lr=0.05,
    )
    verdict = read_guide_verdict(["head_0_1"])

    assert verdict["head_0_1"]["pns"] < 0.3, (
        f"Expected PNS < 0.3 for non-causal head, got {verdict['head_0_1']['pns']:.3f}"
    )


def test_guide_ranks_causal_above_noncausal(planted, x_planted):
    """PNS(head_0_0) > PNS(head_0_1) — guide correctly orders causal relevance."""
    model_fn = _make_model_fn(planted, x_planted)
    cache = run_and_cache(model_fn, sites=["head_0_0", "head_0_1", "logits"])
    alts_h0 = zero_ablation({"head_0_0": cache["head_0_0"]})
    alts_h1 = zero_ablation({"head_0_1": cache["head_0_1"]})

    train_ac_guide(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alts_h0,
        witnesses={},
        consequent_site="logits",
        consequent_value=cache["logits"],
        event_dim=2,
        consequent_scale=1.0,
        n_steps=300,
        lr=0.05,
    )
    pns_h0 = read_guide_verdict(["head_0_0"])["head_0_0"]["pns"]

    train_ac_guide(
        model_fn,
        antecedents={"head_0_1": cache["head_0_1"]},
        alternatives=alts_h1,
        witnesses={},
        consequent_site="logits",
        consequent_value=cache["logits"],
        event_dim=2,
        consequent_scale=1.0,
        n_steps=300,
        lr=0.05,
    )
    pns_h1 = read_guide_verdict(["head_0_1"])["head_0_1"]["pns"]

    assert pns_h0 > pns_h1, (
        f"Expected PNS(h0_0)={pns_h0:.3f} > PNS(h0_1)={pns_h1:.3f}"
    )


def test_guide_is_sampling_succeeds(planted, x_planted):
    """IS estimation completes without NaN and stores a valid PNS in param_store.

    Replaces the SVI loss-decrease test: IS does not train a guide iteratively,
    so there are no losses to decrease. Instead, we verify that the IS sampling
    loop produces n_steps valid log_probs and that read_guide_verdict returns a
    finite PNS for the causal head.
    """
    model_fn = _make_model_fn(planted, x_planted)
    cache = run_and_cache(model_fn, sites=["head_0_0", "logits"])
    alts = zero_ablation({"head_0_0": cache["head_0_0"]})

    _, log_probs = train_ac_guide(
        model_fn,
        antecedents={"head_0_0": cache["head_0_0"]},
        alternatives=alts,
        witnesses={},
        consequent_site="logits",
        consequent_value=cache["logits"],
        event_dim=2,
        consequent_scale=1.0,
        n_steps=100,
        lr=0.05,
    )

    assert len(log_probs) == 100
    # NaN check: NaN != NaN in IEEE 754; any NaN means the IS loop is broken.
    assert all(lp == lp for lp in log_probs), "IS sampling produced NaN log_probs"
    # For the causal head (head_0_0), case=0 necessity world fires with logits→0,
    # giving finite log_prob. At least some samples must be finite.
    assert any(lp > float("-inf") for lp in log_probs), (
        "All log_probs are -inf for causal head — necessity world did not fire"
    )
    verdict = read_guide_verdict(["head_0_0"])
    pns = verdict["head_0_0"]["pns"]
    assert pns == pns, "PNS is NaN after IS estimation"  # NaN check
