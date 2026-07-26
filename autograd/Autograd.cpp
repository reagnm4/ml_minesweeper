#include "Autograd.h"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace nn {

// ---------------------------------------------------------------------------
// Graph plumbing
// ---------------------------------------------------------------------------

Tensor Tensor::constant(Matrix v) {
    Tensor t;
    t.n_ = std::make_shared<Node>();
    t.n_->grad = Matrix(v.rows(), v.cols(), 0.0);
    t.n_->value = std::move(v);
    t.n_->requires_grad = false;
    return t;
}

Tensor Tensor::param(Matrix v) {
    Tensor t;
    t.n_ = std::make_shared<Node>();
    t.n_->grad = Matrix(v.rows(), v.cols(), 0.0);
    t.n_->value = std::move(v);
    t.n_->requires_grad = true;
    return t;
}

Tensor Tensor::make(Matrix value, std::vector<NodeRef> inputs, const char* op,
                    std::function<void(Node&)> backward_fn) {
    Tensor t;
    t.n_ = std::make_shared<Node>();
    t.n_->grad = Matrix(value.rows(), value.cols(), 0.0);
    t.n_->value = std::move(value);
    t.n_->inputs = std::move(inputs);
    t.n_->op = op;
    t.n_->backward_fn = std::move(backward_fn);
    // A result needs a gradient if anything it was built from does.
    for (const NodeRef& in : t.n_->inputs)
        if (in->requires_grad) t.n_->requires_grad = true;
    return t;
}

double Tensor::item() const {
    if (n_->value.size() != 1) throw std::logic_error("item(): tensor is not 1x1");
    return n_->value.flat(0);
}

namespace {

// Post-order over the graph: every node lands in `topo` after all of its
// inputs. Iterative on purpose -- a recursive walk would put the depth of the
// network on the C++ stack.
void build_topo(const NodeRef& root, std::vector<Node*>& topo) {
    std::unordered_set<Node*> visited;
    std::vector<std::pair<Node*, size_t>> stack;

    stack.emplace_back(root.get(), 0);
    visited.insert(root.get());

    while (!stack.empty()) {
        Node* n = stack.back().first;
        const size_t i = stack.back().second;

        if (i < n->inputs.size()) {
            // Advance the cursor *before* pushing: push_back may reallocate,
            // which would invalidate any reference into the stack.
            stack.back().second = i + 1;
            Node* child = n->inputs[i].get();
            if (visited.insert(child).second) stack.emplace_back(child, 0);
            continue;
        }

        topo.push_back(n);
        stack.pop_back();
    }
}

}  // namespace

void Tensor::backward() {
    if (n_->value.size() != 1)
        throw std::logic_error("backward(): can only start from a scalar (1x1) tensor");

    std::vector<Node*> topo;
    build_topo(n_, topo);

    // d(loss)/d(loss) = 1.
    n_->grad.flat(0) = 1.0;

    // Reverse post-order visits every node only after all of its consumers
    // have contributed to its gradient.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn(**it);
}

// ---------------------------------------------------------------------------
// Structural ops
// ---------------------------------------------------------------------------

// C = A B, with A (n x m), B (m x p), C (n x p).
//
//   C_ij = sum_k A_ik B_kj
//
// Writing g_ij = dL/dC_ij and applying the chain rule to every path from
// A_ik to the loss:
//
//   dL/dA_ik = sum_j g_ij dC_ij/dA_ik = sum_j g_ij B_kj = (g B^T)_ik
//   dL/dB_kj = sum_i g_ij dC_ij/dB_kj = sum_i g_ij A_ik = (A^T g)_kj
Tensor matmul(const Tensor& a, const Tensor& b) {
    Matrix out = Matrix::matmul(a.value(), b.value());
    NodeRef an = a.node(), bn = b.node();

    return Tensor::make(std::move(out), {an, bn}, "matmul", [an, bn](Node& self) {
        if (an->requires_grad)
            add_into(an->grad, Matrix::matmul(self.grad, bn->value.transposed()));
        if (bn->requires_grad)
            add_into(bn->grad, Matrix::matmul(an->value.transposed(), self.grad));
    });
}

// out = A + B, elementwise. Each output depends on exactly one element of each
// input with slope 1, so the gradient passes straight through to both.
Tensor add(const Tensor& a, const Tensor& b) {
    Matrix out = a.value() + b.value();
    NodeRef an = a.node(), bn = b.node();

    return Tensor::make(std::move(out), {an, bn}, "add", [an, bn](Node& self) {
        if (an->requires_grad) add_into(an->grad, self.grad);
        if (bn->requires_grad) add_into(bn->grad, self.grad);
    });
}

// out = A * B elementwise, so d(out_ij)/d(A_ij) = B_ij:
//   dL/dA = g (*) B      dL/dB = g (*) A
Tensor mul(const Tensor& a, const Tensor& b) {
    Matrix out = a.value() * b.value();
    NodeRef an = a.node(), bn = b.node();

    return Tensor::make(std::move(out), {an, bn}, "mul", [an, bn](Node& self) {
        if (an->requires_grad) add_into(an->grad, self.grad * bn->value);
        if (bn->requires_grad) add_into(bn->grad, self.grad * an->value);
    });
}

// out = A^T. Element (i,j) of the output is element (j,i) of the input, so the
// incoming gradient just gets transposed back.
Tensor transpose(const Tensor& a) {
    Matrix out = a.value().transposed();
    NodeRef an = a.node();

    return Tensor::make(std::move(out), {an}, "transpose", [an](Node& self) {
        if (an->requires_grad) add_into(an->grad, self.grad.transposed());
    });
}

// out_ij = x_ij + b_0j, i.e. one bias row shared by every sample in the batch.
//
//   dL/dx_ij = g_ij                       (slope 1, one path)
//   dL/db_0j = sum_i g_ij                 (b_0j feeds every row of column j)
//
// The sum over the batch is the whole reason a bias needs its own op: it is
// the transpose of the implicit copy that broadcasting performs.
Tensor add_bias(const Tensor& x, const Tensor& b) {
    if (b.value().rows() != 1 || b.value().cols() != x.value().cols())
        throw std::invalid_argument("add_bias: bias must be 1 x cols(x)");

    Matrix out(x.value().rows(), x.value().cols());
    for (int r = 0; r < out.rows(); ++r)
        for (int c = 0; c < out.cols(); ++c) out(r, c) = x.value()(r, c) + b.value()(0, c);

    NodeRef xn = x.node(), bn = b.node();
    return Tensor::make(std::move(out), {xn, bn}, "add_bias", [xn, bn](Node& self) {
        if (xn->requires_grad) add_into(xn->grad, self.grad);
        if (bn->requires_grad) add_into(bn->grad, sum_rows(self.grad));
    });
}

// out = sum_ij x_ij. Every input contributes with slope 1, so each gets the
// single scalar gradient.
Tensor sum(const Tensor& a) {
    Matrix out(1, 1, a.value().sum());
    NodeRef an = a.node();

    return Tensor::make(std::move(out), {an}, "sum", [an](Node& self) {
        if (!an->requires_grad) return;
        const double g = self.grad.flat(0);
        for (int i = 0; i < an->grad.size(); ++i) an->grad.flat(i) += g;
    });
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

// relu(x) = max(0, x), so the derivative is 1 where x > 0 and 0 where x < 0.
// At exactly x = 0 it is undefined; we take 0, the usual convention. This kink
// is also why gradient checking a ReLU network needs inputs that are not
// sitting within eps of zero.
Tensor relu(const Tensor& x) {
    Matrix out(x.value().rows(), x.value().cols());
    for (int i = 0; i < out.size(); ++i) out.flat(i) = std::max(0.0, x.value().flat(i));

    NodeRef xn = x.node();
    return Tensor::make(std::move(out), {xn}, "relu", [xn](Node& self) {
        if (!xn->requires_grad) return;
        for (int i = 0; i < xn->grad.size(); ++i)
            if (xn->value.flat(i) > 0.0) xn->grad.flat(i) += self.grad.flat(i);
    });
}

// d/dx tanh(x) = 1 - tanh(x)^2, so the forward output is all we need to keep.
Tensor tanh(const Tensor& x) {
    Matrix out(x.value().rows(), x.value().cols());
    for (int i = 0; i < out.size(); ++i) out.flat(i) = std::tanh(x.value().flat(i));

    NodeRef xn = x.node();
    return Tensor::make(std::move(out), {xn}, "tanh", [xn](Node& self) {
        if (!xn->requires_grad) return;
        for (int i = 0; i < xn->grad.size(); ++i) {
            const double y = self.value.flat(i);
            xn->grad.flat(i) += self.grad.flat(i) * (1.0 - y * y);
        }
    });
}

// sigma(x) = 1/(1 + e^-x).
//   sigma'(x) = e^-x / (1 + e^-x)^2 = sigma(x) (1 - sigma(x))
// Again expressible purely in terms of the output.
Tensor sigmoid(const Tensor& x) {
    Matrix out(x.value().rows(), x.value().cols());
    for (int i = 0; i < out.size(); ++i) {
        const double v = x.value().flat(i);
        // Branch on the sign so the exponential never overflows.
        out.flat(i) = v >= 0.0 ? 1.0 / (1.0 + std::exp(-v)) : std::exp(v) / (1.0 + std::exp(v));
    }

    NodeRef xn = x.node();
    return Tensor::make(std::move(out), {xn}, "sigmoid", [xn](Node& self) {
        if (!xn->requires_grad) return;
        for (int i = 0; i < xn->grad.size(); ++i) {
            const double y = self.value.flat(i);
            xn->grad.flat(i) += self.grad.flat(i) * y * (1.0 - y);
        }
    });
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

// L = (1/N) sum_i (p_i - t_i)^2
//   dL/dp_i = (2/N)(p_i - t_i)
Tensor mse_loss(const Tensor& pred, const Matrix& target) {
    if (!pred.value().same_shape(target))
        throw std::invalid_argument("mse_loss: prediction and target shapes differ");

    const int n = pred.value().size();
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = pred.value().flat(i) - target.flat(i);
        total += d * d;
    }

    NodeRef pn = pred.node();
    Matrix t = target;
    return Tensor::make(Matrix(1, 1, total / n), {pn}, "mse_loss",
                        [pn, t, n](Node& self) {
                            if (!pn->requires_grad) return;
                            const double g = self.grad.flat(0);
                            for (int i = 0; i < n; ++i)
                                pn->grad.flat(i) += g * 2.0 * (pn->value.flat(i) - t.flat(i)) / n;
                        });
}

// L = -(1/N) sum_i [ t_i log p_i + (1 - t_i) log(1 - p_i) ]
//
//   dL/dp_i = -(1/N) [ t_i/p_i - (1 - t_i)/(1 - p_i) ]
//           =  (1/N) (p_i - t_i) / (p_i (1 - p_i))
//
// p is clamped away from 0 and 1 so the logarithm and the division stay
// finite. That clamp is exactly why bce_with_logits_loss exists.
namespace {
constexpr double kProbClamp = 1e-12;
double clamp_prob(double p) {
    return std::min(1.0 - kProbClamp, std::max(kProbClamp, p));
}
}  // namespace

Tensor bce_loss(const Tensor& prob, const Matrix& target) {
    if (!prob.value().same_shape(target))
        throw std::invalid_argument("bce_loss: prediction and target shapes differ");

    const int n = prob.value().size();

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double p = clamp_prob(prob.value().flat(i));
        const double t = target.flat(i);
        total -= t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
    }

    NodeRef pn = prob.node();
    Matrix tgt = target;
    return Tensor::make(Matrix(1, 1, total / n), {pn}, "bce_loss",
                        [pn, tgt, n](Node& self) {
                            if (!pn->requires_grad) return;
                            const double g = self.grad.flat(0);
                            for (int i = 0; i < n; ++i) {
                                const double p = clamp_prob(pn->value.flat(i));
                                pn->grad.flat(i) +=
                                    g * (p - tgt.flat(i)) / (p * (1.0 - p) * n);
                            }
                        });
}

// Substituting p = sigma(z) into binary cross-entropy and simplifying:
//
//   L_i = -t log sigma(z) - (1-t) log(1 - sigma(z))
//       = max(z, 0) - z t + log(1 + e^-|z|)
//
// which never evaluates log(0) or overflows exp. The gradient collapses to
// something remarkable -- the p(1-p) from the loss cancels the sigma'(z) from
// the activation:
//
//   dL_i/dz = sigma(z) - t
//
// so with the 1/N averaging, dL/dz = (sigma(z) - t)/N.
Tensor bce_with_logits_loss(const Tensor& logit, const Matrix& target) {
    if (!logit.value().same_shape(target))
        throw std::invalid_argument("bce_with_logits_loss: shapes differ");

    const int n = logit.value().size();
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double z = logit.value().flat(i);
        const double t = target.flat(i);
        total += std::max(z, 0.0) - z * t + std::log1p(std::exp(-std::fabs(z)));
    }

    NodeRef zn = logit.node();
    Matrix tgt = target;
    return Tensor::make(Matrix(1, 1, total / n), {zn}, "bce_with_logits_loss",
                        [zn, tgt, n](Node& self) {
                            if (!zn->requires_grad) return;
                            const double g = self.grad.flat(0);
                            for (int i = 0; i < n; ++i) {
                                const double z = zn->value.flat(i);
                                const double s = z >= 0.0
                                                     ? 1.0 / (1.0 + std::exp(-z))
                                                     : std::exp(z) / (1.0 + std::exp(z));
                                zn->grad.flat(i) += g * (s - tgt.flat(i)) / n;
                            }
                        });
}

}  // namespace nn
