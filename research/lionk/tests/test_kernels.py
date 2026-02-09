from __future__ import annotations

import pytest
import torch

from research.lionk.kernels import (
    power_spectral_update,
    soft_huber_spectral_update,
    zeropower_via_newtonschulz5,
)

# Muon's zeropower_via_newtonschulz5 uses @torch.compile which may fail on
# CPU-only / macOS environments where the inductor backend is unavailable.
# We skip those tests gracefully rather than let them error out.
_compile_works = True
try:
    _test = torch.randn(4, 4)
    zeropower_via_newtonschulz5(_test, steps=1)
except Exception:
    _compile_works = False

requires_compile = pytest.mark.skipif(
    not _compile_works,
    reason="torch.compile inductor backend unavailable (CPU-only / macOS)",
)


# ---------------------------------------------------------------------------
#  Muon (zeropower) kernel
# ---------------------------------------------------------------------------


@requires_compile
def test_muon_kernel_shape_and_finite():
    g = torch.randn(32, 16)
    out = zeropower_via_newtonschulz5(g, steps=2)
    assert out.shape == g.shape
    assert torch.isfinite(out).all()


@requires_compile
def test_muon_kernel_orthogonality():
    """Muon output should be approximately orthonormal: U^T U ~ I."""
    g = torch.randn(8, 32)
    out = zeropower_via_newtonschulz5(g, steps=5)
    # out has shape (8, 32); out @ out^T should be ~ I_8
    gram = out.float() @ out.float().T
    eye = torch.eye(gram.size(0))
    assert torch.allclose(gram, eye, atol=0.1), f"Gram matrix far from I:\n{gram}"


# ---------------------------------------------------------------------------
#  Soft-Huber kernel (family A)
# ---------------------------------------------------------------------------


def test_soft_huber_shape_preserving_for_tall_matrix():
    g = torch.randn(64, 8)
    out = soft_huber_spectral_update(g, delta=0.1, steps=2)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
    assert torch.isfinite(out).all()


def test_soft_huber_shape_preserving_for_wide_matrix():
    g = torch.randn(8, 64)
    out = soft_huber_spectral_update(g, delta=0.1, steps=2)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
    assert torch.isfinite(out).all()


@requires_compile
def test_soft_huber_converges_to_muon_for_small_delta():
    """As delta -> 0, soft-Huber should approach the Muon polar factor."""
    g = torch.randn(16, 32)
    muon_out = zeropower_via_newtonschulz5(g, steps=5).float()
    huber_out = soft_huber_spectral_update(g, delta=1e-6, steps=5).float()
    # They won't be identical (different iteration), but should be close
    cos_sim = torch.nn.functional.cosine_similarity(
        muon_out.flatten().unsqueeze(0),
        huber_out.flatten().unsqueeze(0),
    )
    assert cos_sim.item() > 0.8, f"cos_sim={cos_sim.item():.4f}, expected > 0.8"


# ---------------------------------------------------------------------------
#  Power-compressed kernel (family B)
# ---------------------------------------------------------------------------


def test_power_k_shape_preserving_for_tall_matrix():
    g = torch.randn(64, 8)
    out = power_spectral_update(g, alpha=0.5, delta=0.1)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
    assert torch.isfinite(out).all()


def test_power_k_shape_preserving_for_wide_matrix():
    g = torch.randn(8, 64)
    out = power_spectral_update(g, alpha=0.5, delta=0.1)
    assert out.shape == g.shape
    assert out.dtype == g.dtype
    assert torch.isfinite(out).all()


def test_power_k_alpha1_matches_soft_huber():
    """power_k with alpha=1 should be identical to soft-Huber (same spectral map)."""
    g = torch.randn(16, 32)
    delta = 0.2

    power_out = power_spectral_update(g, alpha=1.0, delta=delta).float()
    # Soft-Huber uses Newton-Schulz which is approximate, power_k uses SVD (exact),
    # so we compare with moderate tolerance.
    huber_out = soft_huber_spectral_update(g, delta=delta, steps=8).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        power_out.flatten().unsqueeze(0),
        huber_out.flatten().unsqueeze(0),
    )
    assert cos_sim.item() > 0.95, f"cos_sim={cos_sim.item():.4f}, expected > 0.95"


def test_power_k_small_alpha_approaches_muon():
    """As alpha -> 0, power_k should approach Muon-like equalization."""
    g = torch.randn(16, 32)
    # Very small alpha with small delta: all singular values mapped close to 1
    out = power_spectral_update(g, alpha=0.01, delta=0.01)
    assert torch.isfinite(out).all()
    # Check that the output has roughly unit-norm rows (like Muon polar factor)
    out_f = out.float()
    norms = out_f.norm(dim=1)
    # All row norms should be similar (equalization)
    assert norms.std() / norms.mean() < 0.3, "Expected equalized row norms for small alpha"


def test_power_k_bounded_singular_values():
    """The spectral map h should produce values in [0, 1]."""
    g = torch.randn(32, 16)
    out = power_spectral_update(g, alpha=0.5, delta=0.1)
    _, S_out, _ = torch.linalg.svd(out.float(), full_matrices=False)
    assert (S_out >= -1e-5).all(), f"Negative singular values: {S_out}"
    assert (S_out <= 1.0 + 1e-5).all(), f"Singular values > 1: {S_out}"


# ---------------------------------------------------------------------------
#  Representative shapes from CifarNet
# ---------------------------------------------------------------------------


def test_kernels_on_representative_shapes():
    """All kernels should work on the actual CifarNet gradient shapes."""
    shapes = [(64, 216), (256, 576), (256, 2304)]
    for m, n in shapes:
        g = torch.randn(m, n)
        assert soft_huber_spectral_update(g, delta=0.1, steps=3).shape == (m, n)
        assert power_spectral_update(g, alpha=0.5, delta=0.1).shape == (m, n)


@requires_compile
def test_muon_on_representative_shapes():
    """Muon kernel should work on the actual CifarNet gradient shapes."""
    shapes = [(64, 216), (256, 576), (256, 2304)]
    for m, n in shapes:
        g = torch.randn(m, n)
        assert zeropower_via_newtonschulz5(g, steps=3).shape == (m, n)
