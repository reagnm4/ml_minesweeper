#include "Layers.h"

#include <cmath>
#include <stdexcept>

namespace nn {

namespace {

// Why initialisation scale matters, and where these numbers come from.
//
// Take a layer y = x W with fan_in inputs, and suppose the inputs and weights
// are independent, zero-mean, with variances Var(x) and Var(W). Each output is
// a sum of fan_in products, so
//
//     Var(y) = fan_in * Var(W) * Var(x)
//
// If fan_in * Var(W) is less than 1, the signal shrinks at every layer and
// vanishes; if it is greater than 1, it explodes. Setting it to 1 keeps the
// forward pass at a constant scale. Running the same argument backwards
// through the chain rule gives fan_out * Var(W) = 1 for the gradient.
//
// Glorot/Xavier compromises between the two, Var(W) = 2/(fan_in + fan_out).
// He notes that ReLU zeroes about half its inputs and so halves the variance,
// and compensates with Var(W) = 2/fan_in.
//
// For a uniform distribution on [-a, a], Var = a^2/3, so a = sqrt(3 Var).
double init_limit(int fan_in, int fan_out, Activation act) {
    if (act == Activation::ReLU) return std::sqrt(6.0 / fan_in);          // sqrt(3 * 2/fan_in)
    return std::sqrt(6.0 / (fan_in + fan_out));  // sqrt(3 * 2/(fan_in+fan_out))
}

Tensor apply(Activation act, const Tensor& z) {
    switch (act) {
        case Activation::ReLU: return relu(z);
        case Activation::Tanh: return nn::tanh(z);
        case Activation::Sigmoid: return sigmoid(z);
        case Activation::None: break;
    }
    return z;
}

}  // namespace

Dense::Dense(int in_features, int out_features, Activation act, std::mt19937_64& rng)
    : act_(act) {
    if (in_features <= 0 || out_features <= 0)
        throw std::invalid_argument("Dense: layer widths must be positive");

    const double limit = init_limit(in_features, out_features, act);
    w_ = Tensor::param(random_uniform(in_features, out_features, rng, -limit, limit));
    // Biases start at zero: they carry no fan-in, so there is no variance
    // argument for randomising them, and zero keeps the first forward pass
    // centred wherever the weights put it.
    b_ = Tensor::param(Matrix(1, out_features, 0.0));
}

Tensor Dense::forward(const Tensor& x) const {
    return apply(act_, add_bias(matmul(x, w_), b_));
}

MLP::MLP(const std::vector<int>& sizes, Activation hidden, Activation output,
         std::mt19937_64& rng) {
    if (sizes.size() < 2) throw std::invalid_argument("MLP: need at least an input and an output");

    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        const bool last = (i + 2 == sizes.size());
        layers_.emplace_back(sizes[i], sizes[i + 1], last ? output : hidden, rng);
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor h = x;
    for (const Dense& layer : layers_) h = layer.forward(h);
    return h;
}

std::vector<Tensor> MLP::parameters() const {
    std::vector<Tensor> out;
    out.reserve(layers_.size() * 2);
    for (const Dense& layer : layers_)
        for (const Tensor& p : layer.parameters()) out.push_back(p);
    return out;
}

void MLP::zero_grad() {
    for (Tensor& p : parameters()) p.zero_grad();
}

}  // namespace nn
