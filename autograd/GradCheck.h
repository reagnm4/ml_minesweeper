// Finite-difference gradient checking -- the project's second, independent
// route to the derivative (spec section 1).
//
// backward() computes the gradient symbolically. This computes it again by
// brute force, nudging one parameter at a time and watching the loss move:
//
//     df/dx  ~=  ( f(x + eps) - f(x - eps) ) / (2 eps)
//
// The central difference is used rather than the forward difference because
// its truncation error is O(eps^2) instead of O(eps): expanding f(x +- eps) as
// a Taylor series, the f''(x) eps^2 terms have the same sign and cancel in the
// subtraction, leaving f'''(x) eps^2 / 6.
//
// If the two routes agree, the calculus is right. That is a statement about
// the math alone, independent of whether any Minesweeper game is ever won.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Autograd.h"

namespace nn {

struct GradCheckReport {
    // Worst relative error over components large enough to measure this way.
    double max_rel = 0.0;
    // Worst *absolute* error over the remaining near-zero components. A
    // relative comparison is meaningless there: eps of 1e-5 divides the
    // roundoff in f (about 1e-16 for doubles) by 2e-5, so a numerical
    // gradient carries roughly 1e-11 of absolute noise no matter what. A
    // component of size 1e-12 is therefore pure noise, and asking for 1e-6
    // relative agreement on it would be asking for noise to match noise.
    double max_abs_small = 0.0;

    // The absolute difference and the gradient magnitude at the point where
    // max_rel occurred. Reported because a relative error is only alarming
    // when the underlying gradient is large: 1e-11 of finite-difference noise
    // sitting on a gradient of 1e-4 reads as 1e-7 relative and is harmless,
    // while the same 1e-7 on a gradient of size 1 would be a real bug.
    double max_rel_diff = 0.0;
    double max_rel_scale = 0.0;

    int components = 0;        // total parameter components checked
    int relative_components = 0;  // how many were big enough for max_rel
    std::string worst;         // where max_rel occurred

    bool passed(double rel_tol = 1e-6, double abs_tol = 1e-9) const {
        return max_rel < rel_tol && max_abs_small < abs_tol;
    }
};

// `forward` must rebuild the graph from the *current* parameter values and
// return the scalar loss. It is called 2 * (number of components) + 1 times,
// so keep the network small.
//
// `magnitude_floor` is the gradient size below which a component is judged by
// absolute rather than relative error.
GradCheckReport grad_check(const std::function<Tensor()>& forward,
                           const std::vector<Tensor>& params, double eps = 1e-5,
                           double magnitude_floor = 1e-4);

}  // namespace nn
