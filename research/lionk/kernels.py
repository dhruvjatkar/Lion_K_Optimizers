"""
Spectral kernels for the Lion-K optimizer family.

In the Lion-K framework (https://www.cs.utexas.edu/~lqiang/lionk/html/intro.html),
choosing a convex spectral K(X) = sum_i g(sigma_i(X)) determines:
  (i)  the nonlinear map applied to momentum via nabla K, and
  (ii) the implicit regularizer/constraint on weights under decoupled weight decay.

Each kernel below implements a different spectral map h(sigma) = g'(sigma),
applied per-singular-value to produce the update U diag(h(sigma_i)) V^T.

Kernel families implemented:
  A. Muon (zeropower):  h(s) = 1.  K = nuclear norm, nabla K = polar factor.
  B. Soft-Huber:        h_d(s) = s / sqrt(s^2 + d^2).  Smooth min(1, s/d).
  C. Power-compressed:  h_{a,d}(s) = s^a / (s^{2a} + d^{2a})^{1/2}.
                         Smooth min(1, (s/d)^a).  Generalises A (a->0) and B (a=1).
"""

from __future__ import annotations

import torch


@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / polar factor of G.

    Spectral map: h(sigma) = 1 for all sigma > 0 (matrix sign / orthogonalization).
    This is the Muon kernel: K(X) = |X|_* (nuclear norm), nabla K = polar factor.
    The implicit constraint is a spectral-norm bound on the weights.

    Compute pattern: polynomial iteration on G G^T applied to G (Muon-style matmuls).
    This function is intentionally copied unchanged from the existing Muon script.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def _apply_shape_preserving(matrix: torch.Tensor, fn):
    """Applies fn to the matrix, transposing if tall so that m <= n inside fn."""
    transposed = matrix.size(0) > matrix.size(1)
    x = matrix.T if transposed else matrix
    y = fn(x)
    return y.T if transposed else y


def _inverse_sqrt_newton_schulz_spd(
    mat: torch.Tensor,
    steps: int,
    eps: float,
) -> torch.Tensor:
    """
    Computes mat^{-1/2} for symmetric positive-definite matrices using
    a fixed-step Newton-Schulz iteration.

    The matrix is normalized by trace/n before iteration for numerical stability.

    Note: unlike zeropower_via_newtonschulz5 (Muon), this is NOT @torch.compile'd.
    Muon's compilation advantage is a real part of its wall-clock story --
    the F0 overhead benchmark measures this asymmetry intentionally.
    """
    n = mat.size(0)
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)

    scale = torch.trace(mat) / float(n)
    scale = torch.clamp(scale, min=eps)
    y = mat / scale
    z = eye.clone()

    for _ in range(steps):
        t = 0.5 * (3.0 * eye - z @ y)
        y = y @ t
        z = t @ z

    return z / torch.sqrt(scale)


def soft_huber_spectral_update(
    G: torch.Tensor,
    delta: float,
    steps: int = 3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Soft-Huber spectral update: (G G^T + delta^2 I)^{-1/2} G.

    Spectral map (Lion-K family A, smooth Huber):
        h_delta(sigma) = sigma / sqrt(sigma^2 + delta^2)

    This is a smooth approximation of h(sigma) = min(1, sigma/delta):
      - Large sigma >> delta:  h -> 1 (Muon-like, full spectral equalization)
      - Small sigma << delta:  h -> sigma/delta (linear, suppresses noisy directions)
      - delta -> 0:  approaches Muon (polar factor, all singular values -> 1)
      - delta -> inf: approaches zero (kills all directions)

    The corresponding convex K is the Huber-style integral:
        g(sigma) = integral_0^sigma h_delta(u) du

    Compute pattern: Newton-Schulz for (A + delta^2 I)^{-1/2} in float32.
    The inner NS iteration is @torch.compile'd for fair F0 benchmarking vs Muon.
    """
    assert len(G.shape) == 2
    delta = max(float(delta), eps)

    def _compute(x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x32 = x.float()
        x32 /= (x32.norm() + eps)
        cov = x32 @ x32.T
        cov = cov + (delta * delta) * torch.eye(
            cov.size(0), device=cov.device, dtype=cov.dtype
        )
        inv_sqrt = _inverse_sqrt_newton_schulz_spd(cov, steps=steps, eps=eps)
        out = inv_sqrt @ x32
        return out.to(x_dtype)

    return _apply_shape_preserving(G, _compute)


def power_spectral_update(
    G: torch.Tensor,
    alpha: float,
    delta: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Power-compressed spectral update (Lion-K family B).

    Spectral map:
        h_{alpha, delta}(sigma) = sigma^alpha / (sigma^{2*alpha} + delta^{2*alpha})^{1/2}

    This is the smooth version of h = min(1, (sigma/delta)^alpha):
      - alpha = 1:  reduces to soft-Huber  h(s) = s / sqrt(s^2 + d^2)
      - alpha < 1:  more aggressive equalization (compresses singular value spread)
      - alpha -> 0: approaches Muon-like constant for all sigma > 0

    The corresponding convex K(X) = sum_i g(sigma_i) with g' = h is well-defined
    for alpha in (0, 1] since h is monotone nondecreasing and bounded in [0, 1].

    Uses SVD for exact spectral map application. Suitable for the moderate-sized
    matrices in CIFAR-10 benchmarks (largest ~256x2304). For large-scale deployment,
    compile to a polynomial/Chebyshev approximation of p_theta(XX^T) X following
    Phase 3 of the research recipe.
    """
    assert len(G.shape) == 2
    delta = max(float(delta), eps)
    alpha = max(min(float(alpha), 1.0), 0.01)  # clamp to avoid 0^0 issues

    def _compute(x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x32 = x.float()
        U, S, Vh = torch.linalg.svd(x32, full_matrices=False)

        # h_{alpha,delta}(sigma) = sigma^alpha / (sigma^{2*alpha} + delta^{2*alpha})^{1/2}
        S_clamped = S.clamp(min=eps)
        S_alpha = S_clamped.pow(alpha)
        delta_alpha = delta**alpha
        h = S_alpha / torch.sqrt(S_alpha.square() + delta_alpha**2)

        out = (U * h.unsqueeze(0)) @ Vh
        return out.to(x_dtype)

    return _apply_shape_preserving(G, _compute)
