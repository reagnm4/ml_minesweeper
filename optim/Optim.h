// Gradient-based optimizers.
//
// Built in the order the spec requires -- plain SGD first, then momentum, then
// Adam -- and each one is derived in a comment above its implementation in
// Optim.cpp before any code appears. tests/gate2a_xor.cpp trains the same
// network with all three so the progression is visible rather than asserted.
#pragma once

#include <vector>

#include "../autograd/Autograd.h"

namespace nn {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    // Apply one update using the gradients currently sitting in the parameters.
    virtual void step() = 0;

    // Gradients accumulate, so this must be called before every backward pass.
    void zero_grad();

    const std::vector<Tensor>& parameters() const { return params_; }

protected:
    std::vector<Tensor> params_;
};

// Stochastic gradient descent, with optional heavy-ball momentum.
// momentum = 0 gives the plain update.
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, double lr, double momentum = 0.0);
    void step() override;

private:
    double lr_;
    double momentum_;
    std::vector<Matrix> velocity_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> params, double lr = 1e-3, double beta1 = 0.9,
         double beta2 = 0.999, double eps = 1e-8);
    void step() override;

private:
    double lr_, beta1_, beta2_, eps_;
    long long t_ = 0;  // step count, for bias correction
    std::vector<Matrix> m_;  // first moment  (mean of the gradient)
    std::vector<Matrix> v_;  // second moment (mean of the squared gradient)
};

}  // namespace nn
