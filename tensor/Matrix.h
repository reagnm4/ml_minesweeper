// Minimal dense 2D matrix -- the only data structure the framework has.
//
// Scope is deliberately tiny (spec section 1 and 6): shape, element access,
// matmul, transpose, and the elementwise arithmetic the chain rule needs.
// No broadcasting engine, no views or strides, no SIMD intrinsics. Those are
// explicitly deferred; adding them now would be generality nothing needs yet.
//
// Everything is double. Float would be faster, but Gate 1 demands agreement
// between analytic and finite-difference gradients to better than 1e-6, and
// single precision cannot deliver that: a central difference divides by
// 2*eps = 2e-5, which amplifies float's ~1e-7 representation error to ~1e-2.
#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace nn {

class Matrix {
public:
    Matrix() = default;
    Matrix(int rows, int cols, double fill = 0.0);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    bool empty() const { return size() == 0; }

    double& operator()(int r, int c) { return data_[static_cast<size_t>(r) * cols_ + c]; }
    double operator()(int r, int c) const { return data_[static_cast<size_t>(r) * cols_ + c]; }

    // Flat access, for loops that do not care about the 2D layout.
    double& flat(int i) { return data_[i]; }
    double flat(int i) const { return data_[i]; }

    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }

    void fill(double v);
    void zero() { fill(0.0); }

    // C = A * B, the ordinary matrix product.
    static Matrix matmul(const Matrix& a, const Matrix& b);

    Matrix transposed() const;

    double sum() const;
    double max_abs() const;

    bool same_shape(const Matrix& o) const { return rows_ == o.rows_ && cols_ == o.cols_; }

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<double> data_;
};

// Elementwise. Shapes must match exactly -- there is no broadcasting.
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);  // Hadamard (elementwise) product
Matrix operator*(double s, const Matrix& a);
Matrix operator*(const Matrix& a, double s);

// In-place accumulate, used everywhere in backward() to sum gradients that
// arrive from several downstream consumers of the same node.
void add_into(Matrix& dst, const Matrix& src);

// Column sums of `a`, returned as a 1 x cols matrix. This is the gradient of
// a bias add with respect to the bias, and the only reduction the framework
// needs.
Matrix sum_rows(const Matrix& a);

// Uniform random fill. Every source of randomness in the framework takes an
// explicit generator so that a whole training run replays from one seed.
Matrix random_uniform(int rows, int cols, std::mt19937_64& rng, double lo, double hi);

}  // namespace nn
