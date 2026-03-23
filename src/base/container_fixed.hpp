#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <cstddef>
#include <span>

namespace DGSEM {
template <class T, std::size_t N>
struct Vec {
  std::array<T, N> data{};

  KOKKOS_INLINE_FUNCTION
  constexpr T &operator[](std::size_t i) noexcept { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator[](std::size_t i) const noexcept {
    return data[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr std::size_t size() noexcept { return N; }

  constexpr std::span<T, N> span() noexcept { return data; }

  constexpr std::span<const T, N> span() const noexcept { return data; }
};

template <class T, std::size_t M, std::size_t N>
struct Mat {
  static_assert(M > 0 && N > 0);

  std::array<T, M * N> data{};

  KOKKOS_INLINE_FUNCTION
  constexpr T &operator()(std::size_t i, std::size_t j) noexcept {
    return data[i * N + j];
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator()(std::size_t i, std::size_t j) const noexcept {
    return data[i * N + j];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr std::size_t rows() noexcept { return M; }

  KOKKOS_INLINE_FUNCTION
  static constexpr std::size_t cols() noexcept { return N; }
  constexpr std::span<T, M * N> span() noexcept { return data; }

  constexpr std::span<const T, M * N> span() const noexcept { return data; }

  KOKKOS_INLINE_FUNCTION
  constexpr Mat<T, N, M> transpose() const noexcept;

  KOKKOS_INLINE_FUNCTION
  constexpr Mat<T, M, N> inverse() const
    requires(M == N);
};

template <class T, std::size_t M, std::size_t N>
KOKKOS_INLINE_FUNCTION constexpr Mat<T, M, N>
operator+(const Mat<T, M, N> &A, const Mat<T, M, N> &B) noexcept {
  Mat<T, M, N> C{};
  for (std::size_t i = 0; i < M * N; ++i)
    C.data[i] = A.data[i] + B.data[i];
  return C;
}

template <class T, std::size_t M, std::size_t N>
KOKKOS_INLINE_FUNCTION constexpr Mat<T, M, N>
operator*(T a, const Mat<T, M, N> &A) noexcept {
  Mat<T, M, N> C{};
  for (std::size_t i = 0; i < M * N; ++i)
    C.data[i] = a * A.data[i];
  return C;
}

template <class T, std::size_t M, std::size_t N>
KOKKOS_INLINE_FUNCTION constexpr Vec<T, M>
operator*(const Mat<T, M, N> &A, const Vec<T, N> &x) noexcept {
  Vec<T, M> y{};
  for (std::size_t i = 0; i < M; ++i) {
    T sum{};
    for (std::size_t j = 0; j < N; ++j)
      sum += A(i, j) * x[j];
    y[i] = sum;
  }
  return y;
}

template <class T, std::size_t M, std::size_t K, std::size_t N>
KOKKOS_INLINE_FUNCTION constexpr Mat<T, M, N>
operator*(const Mat<T, M, K> &A, const Mat<T, K, N> &B) noexcept {
  Mat<T, M, N> C{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      T sum{};
      for (std::size_t k = 0; k < K; ++k)
        sum += A(i, k) * B(k, j);
      C(i, j) = sum;
    }
  return C;
}

template <class T, std::size_t M, std::size_t N>
constexpr Mat<T, N, M> Mat<T, M, N>::transpose() const noexcept {
  Mat<T, N, M> trans{};
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      trans(j, i) = (*this)(i, j);
    }
  return trans;
}

template <class T, std::size_t M, std::size_t N>
constexpr Mat<T, M, N> Mat<T, M, N>::inverse() const
  requires(M == N)
{
  Mat<T, M, N> A = *this;
  Mat<T, M, N> inv{};
  // 初始化为单位阵
  for (std::size_t i = 0; i < M; ++i)
    inv(i, i) = T{1};
  for (std::size_t col = 0; col < N; ++col) {
    // 选主元
    std::size_t pivot = col;
    T max_val = (A(col, col) < 0 ? -A(col, col) : A(col, col));
    for (std::size_t row = col + 1; row < M; ++row) {
      T v = (A(row, col) < 0 ? -A(row, col) : A(row, col));
      if (v > max_val) {
        max_val = v;
        pivot = row;
      }
    }
    // 行交换
    if (pivot != col) {
      for (std::size_t k = 0; k < N; ++k) {
        std::swap(A(col, k), A(pivot, k));
        std::swap(inv(col, k), inv(pivot, k));
      }
    }
    // 主元归一化
    T diag = A(col, col);
    for (std::size_t k = 0; k < N; ++k) {
      A(col, k) /= diag;
      inv(col, k) /= diag;
    }
    // 消元
    for (std::size_t row = 0; row < M; ++row) {
      if (row == col)
        continue;
      T factor = A(row, col);
      for (std::size_t k = 0; k < N; ++k) {
        A(row, k) -= factor * A(col, k);
        inv(row, k) -= factor * inv(col, k);
      }
    }
  }
  return inv;
}
} // namespace DGSEM
