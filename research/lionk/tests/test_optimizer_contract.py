from __future__ import annotations

import pytest
import torch

from research.lionk.config import KConfig
from research.lionk.optimizer import MuonK

# Check whether torch.compile works in this environment
_compile_works = True
try:
    from research.lionk.kernels import zeropower_via_newtonschulz5
    _test = torch.randn(4, 4)
    zeropower_via_newtonschulz5(_test, steps=1)
except Exception:
    _compile_works = False

requires_compile = pytest.mark.skipif(
    not _compile_works,
    reason="torch.compile inductor backend unavailable",
)


def _make_param(shape):
    p = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32))
    p.grad = torch.randn_like(p)
    return p


# ---------------------------------------------------------------------------
#  Muon mode
# ---------------------------------------------------------------------------


@requires_compile
def test_muon_mode_step_updates_state_and_param_shape():
    p = _make_param((16, 8))
    opt = MuonK([p], lr=1e-2, momentum=0.9, nesterov=True, k_config=KConfig(mode="muon"))
    before = p.detach().clone()
    opt.step()
    assert p.shape == before.shape
    assert "momentum_buffer" in opt.state[p]
    assert opt.param_groups[0]["step"] == 1


# ---------------------------------------------------------------------------
#  Soft-Huber mode
# ---------------------------------------------------------------------------


def test_soft_huber_mode_step_runs():
    p = _make_param((8, 4, 3, 3))
    cfg = KConfig(mode="soft_huber_k", delta=0.1, delta_final=0.05, schedule="linear", ns_steps=2)
    opt = MuonK([p], lr=1e-3, momentum=0.5, nesterov=False, k_config=cfg)
    opt.param_groups[0]["total_steps"] = 10
    opt.step()
    assert torch.isfinite(p).all()
    assert opt.param_groups[0]["step"] == 1


# ---------------------------------------------------------------------------
#  Power-K mode
# ---------------------------------------------------------------------------


def test_power_k_mode_step_runs():
    """power_k mode should complete a step without error."""
    p = _make_param((8, 4, 3, 3))
    cfg = KConfig(mode="power_k", delta=0.1, alpha=0.5, ns_steps=2)
    opt = MuonK([p], lr=1e-3, momentum=0.5, nesterov=False, k_config=cfg)
    opt.step()
    assert torch.isfinite(p).all()
    assert opt.param_groups[0]["step"] == 1


def test_power_k_with_schedule():
    """power_k with a cosine delta schedule should run multiple steps."""
    p = _make_param((16, 8))
    cfg = KConfig(
        mode="power_k",
        delta=0.5,
        delta_final=0.01,
        schedule="cosine",
        alpha=0.3,
        ns_steps=2,
    )
    opt = MuonK([p], lr=1e-3, momentum=0.9, nesterov=True, k_config=cfg)
    opt.param_groups[0]["total_steps"] = 20
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert opt.param_groups[0]["step"] == 5


# ---------------------------------------------------------------------------
#  Delta scheduling
# ---------------------------------------------------------------------------


def test_delta_schedule_linear_interpolates():
    """Linear schedule should interpolate between delta and delta_final."""
    cfg = KConfig(mode="soft_huber_k", delta=1.0, delta_final=0.1, schedule="linear")
    opt = MuonK([_make_param((4, 4))], lr=1e-3, k_config=cfg)
    opt.param_groups[0]["total_steps"] = 100

    # At step 0: should be delta (1.0)
    d0 = opt._delta_for_step(0, opt.param_groups[0])
    assert abs(d0 - 1.0) < 1e-5, f"Expected ~1.0, got {d0}"

    # At step 50: should be midpoint (0.55)
    d50 = opt._delta_for_step(50, opt.param_groups[0])
    assert abs(d50 - 0.55) < 0.05, f"Expected ~0.55, got {d50}"

    # At step 100: should be delta_final (0.1)
    d100 = opt._delta_for_step(100, opt.param_groups[0])
    assert abs(d100 - 0.1) < 1e-5, f"Expected ~0.1, got {d100}"


def test_delta_schedule_cosine_endpoints():
    """Cosine schedule should hit endpoints and be smooth in between."""
    cfg = KConfig(mode="soft_huber_k", delta=1.0, delta_final=0.1, schedule="cosine")
    opt = MuonK([_make_param((4, 4))], lr=1e-3, k_config=cfg)
    opt.param_groups[0]["total_steps"] = 100

    d0 = opt._delta_for_step(0, opt.param_groups[0])
    d100 = opt._delta_for_step(100, opt.param_groups[0])
    assert abs(d0 - 1.0) < 1e-5
    assert abs(d100 - 0.1) < 1e-5


# ---------------------------------------------------------------------------
#  Config validation
# ---------------------------------------------------------------------------


def test_power_k_config_validates_alpha():
    """alpha must be in (0, 1] for power_k mode."""
    import pytest

    with pytest.raises(ValueError, match="alpha"):
        KConfig(mode="power_k", alpha=0.0).validate()

    with pytest.raises(ValueError, match="alpha"):
        KConfig(mode="power_k", alpha=1.5).validate()

    # Valid alpha
    cfg = KConfig(mode="power_k", alpha=0.5).validate()
    assert cfg.alpha == 0.5


def test_muon_mode_ignores_alpha():
    """For muon mode, alpha is present but unused -- validation should pass."""
    cfg = KConfig(mode="muon", alpha=0.3).validate()
    assert cfg.mode == "muon"
