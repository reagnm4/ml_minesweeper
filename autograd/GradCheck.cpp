#include "GradCheck.h"

#include <algorithm>
#include <cmath>

namespace nn {

GradCheckReport grad_check(const std::function<Tensor()>& forward,
                           const std::vector<Tensor>& params, double eps,
                           double magnitude_floor) {
    GradCheckReport report;

    // Route 1: one forward pass, one backward pass.
    for (const Tensor& p : params) const_cast<Tensor&>(p).zero_grad();
    Tensor loss = forward();
    loss.backward();

    std::vector<Matrix> analytic;
    analytic.reserve(params.size());
    for (const Tensor& p : params) analytic.push_back(p.grad());

    // Route 2: two extra forward passes per component, no calculus involved.
    for (size_t pi = 0; pi < params.size(); ++pi) {
        Matrix& values = const_cast<Tensor&>(params[pi]).value();

        for (int i = 0; i < values.size(); ++i) {
            const double original = values.flat(i);

            values.flat(i) = original + eps;
            const double up = forward().item();

            values.flat(i) = original - eps;
            const double down = forward().item();

            values.flat(i) = original;  // restore before moving on

            const double numeric = (up - down) / (2.0 * eps);
            const double exact = analytic[pi].flat(i);
            const double diff = std::fabs(exact - numeric);
            const double scale = std::max(std::fabs(exact), std::fabs(numeric));

            ++report.components;
            if (scale >= magnitude_floor) {
                const double rel = diff / scale;
                ++report.relative_components;
                if (rel > report.max_rel) {
                    report.max_rel = rel;
                    report.max_rel_diff = diff;
                    report.max_rel_scale = scale;
                    report.worst = "param " + std::to_string(pi) + " element " +
                                   std::to_string(i) + ": analytic " +
                                   std::to_string(exact) + " vs numeric " +
                                   std::to_string(numeric);
                }
            } else {
                report.max_abs_small = std::max(report.max_abs_small, diff);
            }
        }
    }

    return report;
}

}  // namespace nn
