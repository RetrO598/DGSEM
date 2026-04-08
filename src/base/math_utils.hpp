#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include <type_traits>

namespace DGSEM {

template <typename T>
  requires std::is_floating_point_v<T>
KOKKOS_INLINE_FUNCTION T ln_mean(T x, T y) {
  const T epsilon_f2 = static_cast<T>(1.0e-4);

  T numerator = x * (x - T(2) * y) + y * y;
  T denominator = x * (x + T(2) * y) + y * y;
  T f2 = numerator / denominator;

  if (f2 < epsilon_f2) {
    T poly =
        T(2) + f2 * (T(2.0 / 3.0) + f2 * (T(2.0 / 5.0) + f2 * (T(2.0 / 7.0))));

    return (x + y) / poly;
  } else {
    return (y - x) / std::log(y / x);
  }
}
} // namespace DGSEM