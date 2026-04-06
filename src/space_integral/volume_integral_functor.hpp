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
  using execution_space = typename DataArray::execution_space;
  using team_policy = Kokkos::TeamPolicy<execution_space>;
  using member_type = typename team_policy::member_type;

  VolumeIntegralFunctor(DataArray u_, DataArray du_, const Equations &eq_,
                        const VolumeFlux &flux_,
                        std::array<std::size_t, NDIMS> n_elems_)
      : u(u_), du(du_), eq(eq_), flux(flux_), n_elems(n_elems_) {}

  static void apply(DataArray u_, DataArray du_, const Equations &eq_,
                    const VolumeFlux &flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    VolumeIntegralFunctor functor(u_, du_, eq_, flux_, n_elems_);

    Kokkos::parallel_for("volume_integral", n_elems_[0], functor);
  }

  static void apply(DataArray u_, DataArray du_, const Equations &eq_,
                    const VolumeFlux &flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 2)
  {
    VolumeIntegralFunctor functor(u_, du_, eq_, flux_, n_elems_);

    const auto league_size =
        static_cast<int>(n_elems_[0] * n_elems_[1]);
    Kokkos::parallel_for("volume_integral",
                         team_policy(league_size, Kokkos::AUTO()), functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int &ielem) const
    requires(NDIMS == 1)
  {
    flux(ielem, eq, u, du);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int &ielem,
                                         const int &jelem) const
    requires(NDIMS == 2)
  {
    flux(ielem, jelem, eq, u, du);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const member_type &team) const
    requires(NDIMS == 2)
  {
    const std::size_t league_rank = static_cast<std::size_t>(team.league_rank());
    const std::size_t jelem = league_rank % n_elems[1];
    const std::size_t ielem = league_rank / n_elems[1];
    flux(team, ielem, jelem, eq, u, du);
  }

  DataArray u;
  DataArray du;
  const Equations eq;
  const VolumeFlux flux;
  std::array<std::size_t, NDIMS> n_elems;
};

} // namespace DGSEM
