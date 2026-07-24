#include "Matrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nn {

Matrix::Matrix(int rows, int cols, double fill)
    : rows_(rows), cols_(cols), data_(static_cast<size_t>(rows) * cols, fill) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("Matrix: negative dimension");
}

void Matrix::fill(double v) { std::fill(data_.begin(), data_.end(), v); }

Matrix Matrix::matmul(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows())
        throw std::invalid_argument("matmul: inner dimensions do not match");

    Matrix out(a.rows(), b.cols(), 0.0);

    // i-k-j rather than the textbook i-j-k. Both are the same O(n^3) scalar
    // arithmetic, but this order walks B and the output row-wise, so the inner
    // loop is a contiguous scan instead of a strided one. This is loop
    // ordering, not vectorisation -- SIMD stays deferred (spec section 6).
    for (int i = 0; i < a.rows(); ++i) {
        const double* arow = &a.data()[static_cast<size_t>(i) * a.cols()];
        double* orow = &out.data()[static_cast<size_t>(i) * out.cols()];
        for (int k = 0; k < a.cols(); ++k) {
            const double aik = arow[k];
            if (aik == 0.0) continue;
            const double* brow = &b.data()[static_cast<size_t>(k) * b.cols()];
            for (int j = 0; j < b.cols(); ++j) orow[j] += aik * brow[j];
        }
    }
    return out;
}

Matrix Matrix::transposed() const {
    Matrix out(cols_, rows_);
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) out(c, r) = (*this)(r, c);
    return out;
}

double Matrix::sum() const {
    double s = 0.0;
    for (double v : data_) s += v;
    return s;
}

double Matrix::max_abs() const {
    double m = 0.0;
    for (double v : data_) m = std::max(m, std::fabs(v));
    return m;
}

namespace {
void require_same(const Matrix& a, const Matrix& b, const char* who) {
    if (!a.same_shape(b)) throw std::invalid_argument(std::string(who) + ": shape mismatch");
}
}  // namespace

Matrix operator+(const Matrix& a, const Matrix& b) {
    require_same(a, b, "operator+");
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) + b.flat(i);
    return out;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    require_same(a, b, "operator-");
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) - b.flat(i);
    return out;
}

Matrix operator*(const Matrix& a, const Matrix& b) {
    require_same(a, b, "operator* (Hadamard)");
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) * b.flat(i);
    return out;
}

Matrix operator*(double s, const Matrix& a) {
    Matrix out(a.rows(), a.cols());
    for (int i = 0; i < a.size(); ++i) out.flat(i) = s * a.flat(i);
    return out;
}

Matrix operator*(const Matrix& a, double s) { return s * a; }

void add_into(Matrix& dst, const Matrix& src) {
    require_same(dst, src, "add_into");
    for (int i = 0; i < dst.size(); ++i) dst.flat(i) += src.flat(i);
}

Matrix random_uniform(int rows, int cols, std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    Matrix out(rows, cols);
    for (int i = 0; i < out.size(); ++i) out.flat(i) = dist(rng);
    return out;
}

Matrix sum_rows(const Matrix& a) {
    Matrix out(1, a.cols(), 0.0);
    for (int r = 0; r < a.rows(); ++r)
        for (int c = 0; c < a.cols(); ++c) out(0, c) += a(r, c);
    return out;
}

}  // namespace nn
