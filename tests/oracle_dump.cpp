// Prints this framework's gradients for the exact network in
// tests/oracle/mlp_oracle.py, in the same format, so the two can be diffed.
//
// Run tests/oracle/compare_oracle.sh to do the comparison.
#include <cmath>
#include <cstdio>

#include "../autograd/Autograd.h"
#include "../tensor/Matrix.h"

using namespace nn;

namespace {

constexpr int kBatch = 4, kIn = 3, kHid = 5, kOut = 2;

// Must match gen() in mlp_oracle.py exactly.
Matrix gen(int rows, int cols, double salt) {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) m(i, j) = 0.5 * std::sin(7.0 * i + 13.0 * j + salt);
    return m;
}

void dump(const char* name, const Matrix& g) {
    for (int i = 0; i < g.rows(); ++i)
        for (int j = 0; j < g.cols(); ++j)
            std::printf("%s %d %d %.17g\n", name, i, j, g(i, j));
}

}  // namespace

int main() {
    const Matrix x = gen(kBatch, kIn, 1.0);
    const Matrix y = gen(kBatch, kOut, 2.0);

    Tensor w1 = Tensor::param(gen(kIn, kHid, 3.0));
    Tensor b1 = Tensor::param(gen(1, kHid, 4.0));
    Tensor w2 = Tensor::param(gen(kHid, kOut, 5.0));
    Tensor b2 = Tensor::param(gen(1, kOut, 6.0));

    Tensor h = nn::tanh(add_bias(matmul(Tensor::constant(x), w1), b1));
    Tensor out = add_bias(matmul(h, w2), b2);
    Tensor loss = mse_loss(out, y);
    loss.backward();

    std::printf("loss %.17g\n", loss.item());
    dump("w1", w1.grad());
    dump("b1", b1.grad());
    dump("w2", w2.grad());
    dump("b2", b2.grad());
    return 0;
}
