// Reverse-mode automatic differentiation.
//
// This is the part of the project the whole build order exists to protect.
// Every op below records, at forward time, a closure that knows how to turn
// "the gradient of the loss with respect to my output" into "the gradient of
// the loss with respect to each of my inputs". backward() then walks the graph
// once, from the loss back to the parameters, applying the chain rule.
//
// Each op's derivation is written out above its implementation in
// Autograd.cpp. Read those before trusting anything here; the gradient check
// in tests/gate1_gradcheck.cpp is what proves them.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../tensor/Matrix.h"

namespace nn {

struct Node;
using NodeRef = std::shared_ptr<Node>;

struct Node {
    Matrix value;
    Matrix grad;                 // d(loss)/d(value), accumulated during backward()
    std::vector<NodeRef> inputs;
    std::function<void(Node&)> backward_fn;  // scatters this->grad into inputs' grads
    const char* op = "leaf";
    bool requires_grad = false;
};

// A handle onto a node in the graph. Copying a Tensor shares the node, it does
// not copy the data -- that is what lets an op refer back to its inputs.
class Tensor {
public:
    Tensor() = default;

    // A value the gradient does not flow to: inputs, targets, constants.
    static Tensor constant(Matrix v);
    // A learnable parameter: backward() accumulates into its .grad().
    static Tensor param(Matrix v);

    bool defined() const { return static_cast<bool>(n_); }
    const NodeRef& node() const { return n_; }

    Matrix& value() { return n_->value; }
    const Matrix& value() const { return n_->value; }
    Matrix& grad() { return n_->grad; }
    const Matrix& grad() const { return n_->grad; }

    int rows() const { return n_->value.rows(); }
    int cols() const { return n_->value.cols(); }

    // Scalar shortcut, for losses.
    double item() const;

    void zero_grad() { n_->grad.zero(); }

    // Seed this node's gradient with 1 and propagate to every ancestor.
    // The tensor must be 1x1. Call once per forward graph: intermediate nodes
    // accumulate, so running it twice over the same graph double-counts.
    void backward();

    // Wire up a new node. Public because the ops are free functions.
    static Tensor make(Matrix value, std::vector<NodeRef> inputs, const char* op,
                       std::function<void(Node&)> backward_fn);

private:
    NodeRef n_;
};

// --- structural ops --------------------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);       // same shape, elementwise
Tensor mul(const Tensor& a, const Tensor& b);       // same shape, Hadamard
Tensor transpose(const Tensor& a);

// x is (batch, features), b is (1, features). The one place the framework
// treats a row as if it were repeated down the batch. Written as its own op
// with its own derivative rather than as a general broadcasting rule, which
// the spec defers.
Tensor add_bias(const Tensor& x, const Tensor& b);

// Sum every element into a 1x1 tensor.
Tensor sum(const Tensor& a);

// --- activations -----------------------------------------------------------
Tensor relu(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor sigmoid(const Tensor& x);

// --- losses ----------------------------------------------------------------
// All losses average over every element, so the gradient magnitude does not
// depend on batch size.
Tensor mse_loss(const Tensor& pred, const Matrix& target);

// Expects probabilities already in (0, 1). Kept because it is the textbook
// form and Gate 1 checks it directly; prefer bce_with_logits_loss for
// training.
Tensor bce_loss(const Tensor& prob, const Matrix& target);

// Fused sigmoid + binary cross-entropy. Mathematically identical to
// bce_loss(sigmoid(z), t), but computed in a form that does not overflow or
// take log(0) for large |z|, and whose gradient is simply (sigma(z) - t)/N.
Tensor bce_with_logits_loss(const Tensor& logit, const Matrix& target);

}  // namespace nn
