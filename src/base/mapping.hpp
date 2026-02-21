#pragma once

#include <array>
#include <cstddef>

namespace DGSEM {
template <class T, std::size_t NDIM>
constexpr std::array<T, NDIM>
linear_interpolate(const std::array<T, NDIM> &coord,
                   const std::array<T, NDIM> &left_value,
                   const std::array<T, NDIM> &right_value) {
  std::array<T, NDIM> result{};
  for (std::size_t i = 0; i < NDIM; ++i) {
    result[i] = 0.5 * ((1.0 - coord[i]) * left_value[i] +
                       (1.0 + coord[i]) * right_value[i]);
  }

  return result;
}

template <class T>
constexpr T linear_interpolate(const T &coord, const T &left_value,
                               const T &right_value) {
  return 0.5 * ((1.0 - coord) * left_value + (1.0 + coord) * right_value);
}

template <class T>
class LinearMapping {
public:
  LinearMapping() = default;

  LinearMapping(const T &left, const T &right)
      : left_value(left), right_value(right) {}

  static constexpr T eval(const T &coord, const T &left_value,
                          const T &right_value) {
    return linear_interpolate(coord, left_value, right_value);
  }

  constexpr T eval(const T &coord) const {
    return linear_interpolate(coord, left_value, right_value);
  }

private:
  T left_value{};
  T right_value{};
};

} // namespace DGSEM
