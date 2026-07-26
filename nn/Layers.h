// Dense layer and MLP container.
//
// These are thin: a layer owns its parameters and knows how to build the
// forward graph. All the calculus already lives in autograd/, and has already
// been proven correct by Gate 1.
#pragma once

#include <random>
#include <vector>

#include "../autograd/Autograd.h"

namespace nn {

enum class Activation { None, ReLU, Tanh, Sigmoid };

// y = act(x W + b), with x (batch, in), W (in, out), b (1, out).
class Dense {
public:
    Dense(int in_features, int out_features, Activation act, std::mt19937_64& rng);

    Tensor forward(const Tensor& x) const;

    Tensor& weight() { return w_; }
    Tensor& bias() { return b_; }
    int in_features() const { return w_.rows(); }
    int out_features() const { return w_.cols(); }

    std::vector<Tensor> parameters() const { return {w_, b_}; }

private:
    Tensor w_;
    Tensor b_;
    Activation act_;
};

// A stack of Dense layers. `sizes` lists every layer width including input and
// output, so {2, 4, 1} is a 2-input, one-hidden-layer-of-4, 1-output network.
class MLP {
public:
    MLP(const std::vector<int>& sizes, Activation hidden, Activation output,
        std::mt19937_64& rng);

    Tensor forward(const Tensor& x) const;
    Tensor forward(const Matrix& x) const { return forward(Tensor::constant(x)); }

    std::vector<Tensor> parameters() const;
    void zero_grad();

    const std::vector<Dense>& layers() const { return layers_; }

private:
    std::vector<Dense> layers_;
};

}  // namespace nn
