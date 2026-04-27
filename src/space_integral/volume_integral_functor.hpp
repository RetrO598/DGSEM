#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>

namespace DGSEM {

/**
 * @brief Functor for volume integral in Kokkos parallel loop.
 *
 * @tparam T The value type.
 * @tparam Equations The physical equations type.
 * @tparam Basis The polynomial basis type.
 * @tparam VolumeFlux The volume integral functor type (e.g.
 * VolumeIntegralWeakForm).
 * @tparam NDIMS The number of dimensions.
 */
template <class T, class Equations, class Basis, class VolumeFlux,
          std::size_t NDIMS>
struct VolumeIntegralFunctor {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;

  VolumeIntegralFunctor(DataArray u_, DataArray du_, const Equations& eq_,
                        const VolumeFlux& flux_,
                        std::array<std::size_t, NDIMS> n_elems_)
      : u(u_), du(du_), eq(eq_), flux(flux_), n_elems(n_elems_) {}

  static void apply(DataArray u_, DataArray du_, const Equations& eq_,
                    const VolumeFlux& flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    VolumeIntegralFunctor functor(u_, du_, eq_, flux_, n_elems_);

    Kokkos::parallel_for("volume_integral", n_elems_[0], functor);
  }

  static void apply(DataArray u_, DataArray du_, const Equations& eq_,
                    const VolumeFlux& flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 2)
  {
    VolumeIntegralFunctor functor(u_, du_, eq_, flux_, n_elems_);

    Kokkos::parallel_for("volume_integral",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_elems_[0], n_elems_[1]}),
                         functor);
  }

  static void apply(DataArray u_, DataArray du_, const Equations& eq_,
                    const VolumeFlux& flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 3)
  {
    VolumeIntegralFunctor functor(u_, du_, eq_, flux_, n_elems_);

    Kokkos::parallel_for(
        "volume_integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0], n_elems_[1], n_elems_[2]}),
        functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int& ielem) const
    requires(NDIMS == 1)
  {
    flux(ielem, eq, u, du);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int& ielem,
                                         const int& jelem) const
    requires(NDIMS == 2)
  {
    flux(ielem, jelem, eq, u, du);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int& ielem, const int& jelem,
                                         const int& kelem) const
    requires(NDIMS == 3)
  {
    flux(ielem, jelem, kelem, eq, u, du);
  }

  DataArray u;
  DataArray du;
  const Equations eq;
  const VolumeFlux flux;
  std::array<std::size_t, NDIMS> n_elems;
};

} // namespace DGSEM
