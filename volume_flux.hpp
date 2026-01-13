#pragma once
#include "equations.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct VolumeIntegralWeak;

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 1> {

  template <class Basis, class Equations>
  inline constexpr static void
  weak_form_kernel(std::size_t ielem, const Equations &eq,
                   const xt::xarray<T> &u, xt::xarray<T> &du) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      std::array<T, NVARS> flux;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }
      flux = eq.flux(u_node, 0);

      for (std::size_t iinode = 0; iinode < Basis::NNodes; ++iinode) {
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, iinode, var) =
              du(ielem, iinode, var) +
              Basis::derivative_dhat(iinode, inode) * flux[var];
        }
      }
    }
  }
};

} // namespace detail

template <class Basis, class Equations>
struct VolumeIntegralWeakForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr void operator()(std::size_t ielem, const Equations &eq,
                                   const xt::xarray<value_type> &u,
                                   xt::xarray<value_type> &du) {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel<Basis, Equations>(ielem, eq, u, du);
  }
};

} // namespace DGSEM