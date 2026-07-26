#include "Optim.h"

#include <cmath>

namespace nn {

void Optimizer::zero_grad() {
    for (Tensor& p : params_) p.zero_grad();
}

// ---------------------------------------------------------------------------
// 1. Plain gradient descent
//
// The gradient g = dL/dtheta points in the direction of steepest *increase* of
// the loss, so stepping against it decreases the loss. Formally, a first-order
// Taylor expansion around the current parameters gives
//
//     L(theta - lr*g) ~= L(theta) - lr * g.g = L(theta) - lr * ||g||^2
//
// The correction is negative for any lr > 0, so for a small enough step the
// loss is guaranteed to fall. "Small enough" is doing real work in that
// sentence: the expansion drops an O(lr^2) term proportional to the curvature,
// and once lr grows past roughly 2/(largest curvature) the step overshoots the
// valley and the loss diverges. That is the whole learning-rate story.
//
//     theta <- theta - lr * g
//
// 2. Momentum (Polyak's heavy ball)
//
// Plain descent has a specific failure: in a ravine -- steep across, shallow
// along -- the gradient mostly points across the ravine, so the iterates
// bounce between the walls and crawl along the floor. Momentum keeps a running
// velocity instead of using the raw gradient:
//
//     v <- mu*v + g
//     theta <- theta - lr * v
//
// Unrolling that recursion, v_t = sum_i mu^(t-i) g_i, an exponentially
// weighted sum of past gradients. The across-ravine components alternate in
// sign and cancel in the sum; the along-ravine component is consistent and
// accumulates, converging to g/(1 - mu). So momentum damps the oscillation and
// multiplies the effective step along any consistent direction by 1/(1 - mu)
// -- a factor of 10 at the usual mu = 0.9.
// ---------------------------------------------------------------------------

SGD::SGD(std::vector<Tensor> params, double lr, double momentum)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum) {
    if (momentum_ != 0.0) {
        velocity_.reserve(params_.size());
        for (const Tensor& p : params_)
            velocity_.emplace_back(p.value().rows(), p.value().cols(), 0.0);
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        Matrix& theta = params_[i].value();
        const Matrix& g = params_[i].grad();

        if (momentum_ == 0.0) {
            for (int k = 0; k < theta.size(); ++k) theta.flat(k) -= lr_ * g.flat(k);
            continue;
        }

        Matrix& v = velocity_[i];
        for (int k = 0; k < theta.size(); ++k) {
            v.flat(k) = momentum_ * v.flat(k) + g.flat(k);
            theta.flat(k) -= lr_ * v.flat(k);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Adam (adaptive moment estimation)
//
// Momentum fixes the *direction* problem but not the *scale* problem: one
// global learning rate has to suit every parameter, even though gradients for
// different weights can differ by orders of magnitude. Adam gives each
// parameter its own effective step by tracking two exponential moving
// averages -- of the gradient, and of its square:
//
//     m <- b1*m + (1 - b1)*g          (estimate of the mean of g)
//     v <- b2*v + (1 - b2)*g^2        (estimate of the uncentred variance)
//
// and then dividing the mean by the root of the second moment:
//
//     theta <- theta - lr * m / (sqrt(v) + eps)
//
// The ratio is dimensionless -- scaling every gradient by c scales m by c and
// sqrt(v) by c, leaving the step unchanged -- so the update is roughly lr in
// size for every parameter regardless of its gradient scale. A parameter whose
// gradient is consistent gets |m|/sqrt(v) near 1 and moves a full step; one
// whose gradient keeps flipping sign averages to m near 0 and barely moves.
//
// Bias correction. Both averages start at zero, which drags the early
// estimates toward zero. Unrolling the first:
//
//     m_t = (1 - b1) * sum_{i=1..t} b1^(t-i) g_i
//
// If the gradient were a constant g, the geometric sum gives
// E[m_t] = g * (1 - b1^t), i.e. the estimate is short by exactly the factor
// (1 - b1^t). Dividing it out removes the bias, and matters most at small t:
// at t = 1 with b1 = 0.9 the raw estimate is 10% of the true gradient.
//
//     m_hat = m / (1 - b1^t)          v_hat = v / (1 - b2^t)
//
// eps only keeps the division finite when v_hat is zero.
// ---------------------------------------------------------------------------

Adam::Adam(std::vector<Tensor> params, double lr, double beta1, double beta2, double eps)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const Tensor& p : params_) {
        m_.emplace_back(p.value().rows(), p.value().cols(), 0.0);
        v_.emplace_back(p.value().rows(), p.value().cols(), 0.0);
    }
}

void Adam::step() {
    ++t_;
    const double bias1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
    const double bias2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        Matrix& theta = params_[i].value();
        const Matrix& g = params_[i].grad();
        Matrix& m = m_[i];
        Matrix& v = v_[i];

        for (int k = 0; k < theta.size(); ++k) {
            const double gk = g.flat(k);
            m.flat(k) = beta1_ * m.flat(k) + (1.0 - beta1_) * gk;
            v.flat(k) = beta2_ * v.flat(k) + (1.0 - beta2_) * gk * gk;

            const double m_hat = m.flat(k) / bias1;
            const double v_hat = v.flat(k) / bias2;
            theta.flat(k) -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

}  // namespace nn
