#pragma once
#include <array>
#include <cstddef>
#include <span>

namespace DGSEM {
template <class T, std::size_t N>
struct Vec {
  std::array<T, N> data{};

  constexpr T &operator[](std::size_t i) noexcept { return data[i]; }

  constexpr const T &operator[](std::size_t i) const noexcept {
    return data[i];
  }

  static constexpr std::size_t size() noexcept { return N; }

  constexpr std::span<T, N> span() noexcept { return data; }

  constexpr std::span<const T, N> span() const noexcept { return data; }
};

template <class T, std::size_t M, std::size_t N>
struct Mat {
  static_assert(M > 0 && N > 0);

  std::array<T, M * N> data{};
  constexpr T &operator()(std::size_t i, std::size_t j) noexcept {
    return data[i * N + j];
  }

  constexpr const T &operator()(std::size_t i, std::size_t j) const noexcept {
    return data[i * N + j];
  }
  static constexpr std::size_t rows() noexcept { return M; }
  static constexpr std::size_t cols() noexcept { return N; }
  constexpr std::span<T, M * N> span() noexcept { return data; }

  constexpr std::span<const T, M * N> span() const noexcept { return data; }

  constexpr Mat<T, N, M> transpose() const noexcept;
};

template <class T, std::size_t M, std::size_t N>
constexpr Mat<T, M, N> operator+(const Mat<T, M, N> &A,
                                 const Mat<T, M, N> &B) noexcept {
  Mat<T, M, N> C{};
  for (std::size_t i = 0; i < M * N; ++i)
    C.data[i] = A.data[i] + B.data[i];
  return C;
}

template <class T, std::size_t M, std::size_t N>
constexpr Mat<T, M, N> operator*(T a, const Mat<T, M, N> &A) noexcept {
  Mat<T, M, N> C{};
  for (std::size_t i = 0; i < M * N; ++i)
    C.data[i] = a * A.data[i];
  return C;
}

template <class T, std::size_t M, std::size_t N>
constexpr Vec<T, M> operator*(const Mat<T, M, N> &A,
                              const Vec<T, N> &x) noexcept {
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
constexpr Mat<T, M, N> operator*(const Mat<T, M, K> &A,
                                 const Mat<T, K, N> &B) noexcept {
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
} // namespace DGSEM
