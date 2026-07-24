#!/usr/bin/env python3
"""Throwaway NumPy oracle -- spec Stage 1, the 'recommended' cross-check.

This is the ONLY place NumPy is allowed (spec section 2). It is not part of the
framework and nothing links against it.

It implements the same 3 -> 5 -> 2 tanh MLP with MSE loss twice over:
forward/backward written out by hand, and a finite-difference check of that
backward. Then it prints the gradients so they can be compared against the C++
implementation's, giving a third independent route to the same numbers --
different language, different code, same calculus.

Parameters are generated from a closed form rather than a random seed so that
the C++ side can reproduce exactly the same values without serialising
anything.
"""

import numpy as np

BATCH, N_IN, N_HID, N_OUT = 4, 3, 5, 2


def gen(rows, cols, salt):
    """Deterministic fill, reproducible in any language with sin()."""
    i = np.arange(rows).reshape(-1, 1)
    j = np.arange(cols).reshape(1, -1)
    return 0.5 * np.sin(7.0 * i + 13.0 * j + salt)


X = gen(BATCH, N_IN, 1.0)
Y = gen(BATCH, N_OUT, 2.0)
W1, B1 = gen(N_IN, N_HID, 3.0), gen(1, N_HID, 4.0)
W2, B2 = gen(N_HID, N_OUT, 5.0), gen(1, N_OUT, 6.0)


def forward(w1, b1, w2, b2):
    z1 = X @ w1 + b1          # bias broadcast down the batch
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    loss = np.mean((z2 - Y) ** 2)
    return loss, (z1, a1, z2)


def backward(w1, b1, w2, b2):
    loss, (z1, a1, z2) = forward(w1, b1, w2, b2)
    n = z2.size
    gz2 = 2.0 * (z2 - Y) / n           # d(mean square error)/d(z2)
    gw2 = a1.T @ gz2                   # matmul rule: dL/dB = A^T g
    gb2 = gz2.sum(axis=0, keepdims=True)   # bias gradient sums over the batch
    ga1 = gz2 @ w2.T                   # matmul rule: dL/dA = g B^T
    gz1 = ga1 * (1.0 - a1 ** 2)        # tanh'(z) = 1 - tanh(z)^2
    gw1 = X.T @ gz1
    gb1 = gz1.sum(axis=0, keepdims=True)
    return loss, [gw1, gb1, gw2, gb2]


def finite_difference(params, eps=1e-5):
    """Central difference over every element of every parameter."""
    out = []
    for k, p in enumerate(params):
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + eps
            up = forward(*params)[0]
            p[idx] = original - eps
            down = forward(*params)[0]
            p[idx] = original
            g[idx] = (up - down) / (2.0 * eps)
        out.append(g)
    return out


def main():
    params = [W1, B1, W2, B2]
    loss, analytic = backward(*params)
    numeric = finite_difference(params)

    worst = 0.0
    for a, n in zip(analytic, numeric):
        scale = np.maximum(np.abs(a), np.abs(n))
        mask = scale >= 1e-4
        if mask.any():
            worst = max(worst, float((np.abs(a - n)[mask] / scale[mask]).max()))
    print(f"# numpy self-check: max relative error {worst:.3e}", flush=True)

    print(f"loss {loss:.17g}")
    for name, g in zip(["w1", "b1", "w2", "b2"], analytic):
        for idx in np.ndindex(g.shape):
            print(f"{name} {idx[0]} {idx[1]} {g[idx]:.17g}")


if __name__ == "__main__":
    main()
