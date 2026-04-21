#pragma once

#include <Kokkos_Macros.hpp>
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <type_traits>

namespace DGSEM::utils {

namespace detail {
template <class T, std::size_t NDIMS>
KOKKOS_INLINE_FUNCTION std::array<T, NDIMS + 2>
prim_to_cons_impl(const std::array<T, NDIMS + 2>& prim, T gamma) {
  std::array<T, NDIMS + 2> cons{};
  cons[0] = prim[0];

  T velocity_sq = T{0};
  for (std::size_t dim = 0; dim < NDIMS; ++dim) {
    const T velocity = prim[dim + 1];
    cons[dim + 1] = prim[0] * velocity;
    velocity_sq += velocity * velocity;
  }

  cons[NDIMS + 1] = prim[NDIMS + 1] / (gamma - static_cast<T>(1)) +
                    static_cast<T>(0.5) * prim[0] * velocity_sq;
  return cons;
}

template <class T, std::size_t NDIMS>
KOKKOS_INLINE_FUNCTION std::array<T, NDIMS + 2>
cons_to_prim_impl(const std::array<T, NDIMS + 2>& cons, T gamma) {
  std::array<T, NDIMS + 2> prim{};
  prim[0] = cons[0];

  T velocity_sq = T{0};
  for (std::size_t dim = 0; dim < NDIMS; ++dim) {
    const T velocity = cons[dim + 1] / cons[0];
    prim[dim + 1] = velocity;
    velocity_sq += velocity * velocity;
  }

  prim[NDIMS + 1] =
      (gamma - static_cast<T>(1)) *
      (cons[NDIMS + 1] - static_cast<T>(0.5) * cons[0] * velocity_sq);
  return prim;
}

template <class T, std::size_t NDIMS>
KOKKOS_INLINE_FUNCTION std::array<T, NDIMS + 2>
cons_to_prim_impl(const Kokkos::Array<T, NDIMS + 2>& cons, T gamma) {
  std::array<T, NDIMS + 2> prim{};
  prim[0] = cons[0];

  T velocity_sq = T{0};
  for (std::size_t dim = 0; dim < NDIMS; ++dim) {
    const T velocity = cons[dim + 1] / cons[0];
    prim[dim + 1] = velocity;
    velocity_sq += velocity * velocity;
  }

  prim[NDIMS + 1] =
      (gamma - static_cast<T>(1)) *
      (cons[NDIMS + 1] - static_cast<T>(0.5) * cons[0] * velocity_sq);
  return prim;
}

template <class Equations>
using equation_traits = equations::EquationTraits<Equations>;
} // namespace detail

template <class T, std::size_t N>
KOKKOS_INLINE_FUNCTION std::array<T, N>
prim_to_cons(const std::array<T, N>& prim, T gamma) {
  static_assert(N >= 3, "prim_to_cons expects at least 1D Euler state data.");
  return detail::prim_to_cons_impl<T, N - 2>(prim, gamma);
}

template <class T, std::size_t N>
KOKKOS_INLINE_FUNCTION std::array<T, N>
cons_to_prim(const std::array<T, N>& cons, T gamma) {
  static_assert(N >= 3, "cons_to_prim expects at least 1D Euler state data.");
  return detail::cons_to_prim_impl<T, N - 2>(cons, gamma);
}

template <class T, std::size_t N>
KOKKOS_INLINE_FUNCTION std::array<T, N>
cons_to_prim(const Kokkos::Array<T, N>& cons, T gamma) {
  static_assert(N >= 3, "cons_to_prim expects at least 1D Euler state data.");
  return detail::cons_to_prim_impl<T, N - 2>(cons, gamma);
}

template <class Equations>
  requires std::is_class_v<Equations>
KOKKOS_INLINE_FUNCTION auto prim_to_cons(
    const std::array<typename detail::equation_traits<Equations>::value_type,
                     detail::equation_traits<Equations>::NVARS>& prim,
    const Equations& eq)
    -> std::array<typename detail::equation_traits<Equations>::value_type,
                  detail::equation_traits<Equations>::NVARS> {
  using traits = detail::equation_traits<Equations>;
  using T = typename traits::value_type;
  static_assert(traits::NVARS == traits::NDIMS + 2,
                "prim_to_cons expects an Euler-like state layout.");
  return detail::prim_to_cons_impl<T, traits::NDIMS>(prim, eq.get_gamma());
}

template <class Equations>
  requires std::is_class_v<Equations>
KOKKOS_INLINE_FUNCTION auto cons_to_prim(
    const std::array<typename detail::equation_traits<Equations>::value_type,
                     detail::equation_traits<Equations>::NVARS>& cons,
    const Equations& eq)
    -> std::array<typename detail::equation_traits<Equations>::value_type,
                  detail::equation_traits<Equations>::NVARS> {
  using traits = detail::equation_traits<Equations>;
  using T = typename traits::value_type;
  static_assert(traits::NVARS == traits::NDIMS + 2,
                "cons_to_prim expects an Euler-like state layout.");
  return detail::cons_to_prim_impl<T, traits::NDIMS>(cons, eq.get_gamma());
}

} // namespace DGSEM::utils
