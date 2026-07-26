// VERIFICATION GATE 1 (spec section 3, Stage 1) -- MANDATORY, BLOCKING.
//
//   "For every op and for a small end-to-end network, compare the analytic
//    gradient from backward() against a numerical finite-difference gradient
//    (f(x+eps) - f(x-eps)) / (2*eps), eps ~= 1e-5. Max relative error must be
//    < 1e-6. No later stage begins until this passes for every parameter."
//
// Each op is checked in isolation first, so a failure names the op rather than
// "the network". Every op is reduced to a scalar with sum(out * W) for a fixed
// random constant W, never a plain sum(out): a plain sum sends an all-ones
// gradient backwards, which happily hides a transposed or symmetric mistake.
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "../autograd/Autograd.h"
#include "../autograd/GradCheck.h"
#include "../tensor/Matrix.h"
#include "TestUtil.h"

using namespace nn;
using test::check;
using test::note;
using test::section;

namespace {

constexpr double kRelTol = 1e-6;   // the spec's threshold
constexpr double kAbsTol = 1e-9;   // for components too small to compare relatively
constexpr double kEps = 1e-5;

std::mt19937_64 rng(0xC0FFEE);

double g_worst_rel = 0.0;
double g_worst_abs = 0.0;

// A fixed random weighting matrix, built ONCE per check and captured by the
// forward closure. It must not be regenerated inside the closure: grad_check
// calls forward() repeatedly and differences the results, so a weighting that
// changed between calls would mean differencing two different functions.
Tensor weights(int rows, int cols) {
    return Tensor::constant(random_uniform(rows, cols, rng, -1.5, 1.5));
}

void run(const std::string& name, const std::function<Tensor()>& forward,
         const std::vector<Tensor>& params) {
    const GradCheckReport r = grad_check(forward, params, kEps);
    g_worst_rel = std::max(g_worst_rel, r.max_rel);
    g_worst_abs = std::max(g_worst_abs, r.max_abs_small);

    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "max rel %.2e (abs %.1e on a gradient of %.1e) over %d/%d components",
                  r.max_rel, r.max_rel_diff, r.max_rel_scale, r.relative_components,
                  r.components);
    check(r.passed(kRelTol, kAbsTol), name, buf);
    if (!r.passed(kRelTol, kAbsTol) && !r.worst.empty()) note("worst: " + r.worst);
}

Tensor P(int rows, int cols, double lo = -1.5, double hi = 1.5) {
    return Tensor::param(random_uniform(rows, cols, rng, lo, hi));
}

Matrix R(int rows, int cols, double lo = -1.0, double hi = 1.0) {
    return random_uniform(rows, cols, rng, lo, hi);
}

Matrix binary_targets(int rows, int cols) {
    Matrix m(rows, cols);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < m.size(); ++i) m.flat(i) = coin(rng);
    return m;
}

}  // namespace

int main() {
    section("structural ops");
    {
        Tensor a = P(4, 3), b = P(3, 5);
        Tensor w = weights(4, 5);
        run("matmul: d/dA and d/dB", [&] { return sum(mul(matmul(a, b), w)); }, {a, b});
    }
    {
        Tensor a = P(4, 3), b = P(4, 3);
        Tensor w = weights(4, 3);
        run("add: d/dA and d/dB", [&] { return sum(mul(add(a, b), w)); }, {a, b});
    }
    {
        Tensor a = P(4, 3), b = P(4, 3);
        Tensor w = weights(4, 3);
        run("mul (Hadamard): d/dA and d/dB", [&] { return sum(mul(mul(a, b), w)); }, {a, b});
    }
    {
        Tensor a = P(4, 3);
        Tensor w = weights(3, 4);
        run("transpose: d/dA", [&] { return sum(mul(transpose(a), w)); }, {a});
    }
    {
        Tensor x = P(6, 4), bias = P(1, 4);
        Tensor w = weights(6, 4);
        run("add_bias: d/dx and d/dbias (the batch sum)",
            [&] { return sum(mul(add_bias(x, bias), w)); }, {x, bias});
    }
    {
        Tensor a = P(4, 3);
        run("sum: d/dA", [&] { return sum(a); }, {a});
    }

    section("activations");
    {
        Tensor x = P(5, 4, -2.0, 2.0);
        Tensor w = weights(5, 4);
        run("relu: d/dx", [&] { return sum(mul(relu(x), w)); }, {x});
    }
    {
        Tensor x = P(5, 4, -2.0, 2.0);
        Tensor w = weights(5, 4);
        run("tanh: d/dx", [&] { return sum(mul(nn::tanh(x), w)); }, {x});
    }
    {
        Tensor x = P(5, 4, -3.0, 3.0);
        Tensor w = weights(5, 4);
        run("sigmoid: d/dx", [&] { return sum(mul(sigmoid(x), w)); }, {x});
    }

    section("losses");
    {
        Tensor pred = P(6, 3);
        const Matrix target = R(6, 3);
        run("mse_loss: d/dpred", [&] { return mse_loss(pred, target); }, {pred});
    }
    {
        // Probabilities kept clear of 0 and 1 so the +-eps nudge stays legal.
        Tensor prob = P(6, 3, 0.08, 0.92);
        const Matrix target = binary_targets(6, 3);
        run("bce_loss: d/dprob", [&] { return bce_loss(prob, target); }, {prob});
    }
    {
        Tensor logit = P(6, 3, -3.0, 3.0);
        const Matrix target = binary_targets(6, 3);
        run("bce_with_logits_loss: d/dlogit",
            [&] { return bce_with_logits_loss(logit, target); }, {logit});
    }

    section("gradient accumulation at a fan-out");
    {
        // h is consumed twice. If backward() overwrote gradients instead of
        // accumulating them, this is where it would show up.
        Tensor x = P(4, 3);
        Tensor w = weights(4, 3);
        run("a node used by two consumers accumulates both gradients",
            [&] {
                Tensor h = nn::tanh(x);
                return sum(mul(add(mul(h, h), h), w));
            },
            {x});
    }
    {
        Tensor v = P(3, 3);
        Tensor w = weights(3, 3);
        run("a parameter reused at two depths",
            [&] {
                Tensor h = matmul(v, v);
                return sum(mul(matmul(h, v), w));
            },
            {v});
    }

    section("end-to-end networks");
    {
        // 3 -> 5 -> 4 -> 2, tanh hidden, MSE output. Batch of 6.
        Tensor w1 = P(3, 5), b1 = P(1, 5);
        Tensor w2 = P(5, 4), b2 = P(1, 4);
        Tensor w3 = P(4, 2), b3 = P(1, 2);
        const Matrix x = R(6, 3);
        const Matrix y = R(6, 2);

        run("MLP 3-5-4-2, tanh, MSE: every weight and bias",
            [&] {
                Tensor h = Tensor::constant(x);
                h = nn::tanh(add_bias(matmul(h, w1), b1));
                h = nn::tanh(add_bias(matmul(h, w2), b2));
                h = add_bias(matmul(h, w3), b3);
                return mse_loss(h, y);
            },
            {w1, b1, w2, b2, w3, b3});
    }
    {
        Tensor w1 = P(3, 6), b1 = P(1, 6);
        Tensor w2 = P(6, 1), b2 = P(1, 1);
        const Matrix x = R(8, 3);
        const Matrix y = binary_targets(8, 1);

        run("MLP 3-6-1, ReLU, sigmoid + BCE: every weight and bias",
            [&] {
                Tensor h = Tensor::constant(x);
                h = relu(add_bias(matmul(h, w1), b1));
                h = sigmoid(add_bias(matmul(h, w2), b2));
                return bce_loss(h, y);
            },
            {w1, b1, w2, b2});

        run("MLP 3-6-1, ReLU, fused BCE-with-logits: every weight and bias",
            [&] {
                Tensor h = Tensor::constant(x);
                h = relu(add_bias(matmul(h, w1), b1));
                h = add_bias(matmul(h, w2), b2);
                return bce_with_logits_loss(h, y);
            },
            {w1, b1, w2, b2});
    }

    section("cross-checks between two routes to the same quantity");
    {
        // bce(sigmoid(z), t) and bce_with_logits(z, t) are the same function
        // written two ways. Both value and gradient must agree.
        Tensor z1 = Tensor::param(random_uniform(5, 3, rng, -4.0, 4.0));
        Tensor z2 = Tensor::param(z1.value());
        const Matrix t = binary_targets(5, 3);

        Tensor l1 = bce_loss(sigmoid(z1), t);
        l1.backward();
        Tensor l2 = bce_with_logits_loss(z2, t);
        l2.backward();

        const double dval = std::fabs(l1.item() - l2.item());
        double dgrad = 0.0;
        for (int i = 0; i < z1.grad().size(); ++i)
            dgrad = std::max(dgrad, std::fabs(z1.grad().flat(i) - z2.grad().flat(i)));

        char buf[160];
        std::snprintf(buf, sizeof buf, "loss differs by %.3e, gradient by %.3e", dval, dgrad);
        check(dval < 1e-12 && dgrad < 1e-12, "sigmoid+BCE equals fused BCE-with-logits", buf);
    }
    {
        // A case with a gradient that can be written down by hand:
        // f(w) = sum((x w)^2)/N with x, w scalars-in-a-1x1 gives df/dw = 2 x^2 w / 1.
        Tensor w = Tensor::param(Matrix(1, 1, 0.7));
        const Matrix x(1, 1, 1.3);
        const Matrix zero(1, 1, 0.0);
        Tensor pred = matmul(Tensor::constant(x), w);
        Tensor loss = mse_loss(pred, zero);
        loss.backward();

        const double expected = 2.0 * 1.3 * 1.3 * 0.7;
        char buf[160];
        std::snprintf(buf, sizeof buf, "analytic %.15f vs hand-derived %.15f", w.grad().flat(0),
                      expected);
        check(std::fabs(w.grad().flat(0) - expected) < 1e-15,
              "backward() matches a gradient derived by hand", buf);
    }

    section("gate 1 verdict");
    {
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "worst relative error across every op and network: %.3e (threshold %.0e)",
                      g_worst_rel, kRelTol);
        check(g_worst_rel < kRelTol, "GATE 1: max relative error < 1e-6", buf);
        std::snprintf(buf, sizeof buf, "worst absolute error on near-zero components: %.3e",
                      g_worst_abs);
        check(g_worst_abs < kAbsTol, "GATE 1: near-zero components agree absolutely", buf);
    }

    return test::summary("gate 1");
}
