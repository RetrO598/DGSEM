#pragma once

#include "base/global_def.hpp"
#include <Kokkos_Macros.hpp>
#include <array>
#include <cstddef>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <traits/Kokkos_IterationPatternTrait.hpp>
namespace DGSEM {
template <class T, class Functor, std::size_t NDIMS>
struct InitialFunctor {
  using DataArray = solution_type_traits<T, NDIMS>::DataArray;
  using CoordArray = typename coordinate_type_traits<T, NDIMS>::CoordArray;

  InitialFunctor(DataArray u_, CoordArray node_coords_, Functor initial_,
                 std::array<std::size_t, NDIMS> n_elems_, std::size_t n_dofs_,
                 std::size_t NVARS_)
      : u(u_), node_coords(node_coords_), initial(initial_), n_elems(n_elems_),
        n_dofs(n_dofs_), NVARS(NVARS_) {}

  static void apply(DataArray u_, CoordArray node_coords_, Functor initial_,
                    std::array<std::size_t, NDIMS> n_elems_,
                    std::size_t n_dofs_, std::size_t NVARS_)
    requires(NDIMS == 1)
  {
    InitialFunctor<T, Functor, NDIMS> functor(u_, node_coords_, initial_,
                                              n_elems_, n_dofs_, NVARS_);

    Kokkos::parallel_for("initialize_elements", n_elems_[0], functor);
  }

  static void apply(DataArray u_, CoordArray node_coords_, Functor initial_,
                    std::array<std::size_t, NDIMS> n_elems_,
                    std::size_t n_dofs_, std::size_t NVARS_)
    requires(NDIMS == 2)
  {
    InitialFunctor<T, Functor, NDIMS> functor(u_, node_coords_, initial_,
                                              n_elems_, n_dofs_, NVARS_);

    Kokkos::parallel_for("initialize_elements",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_elems_[0], n_elems_[1]}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int& elem) const
    requires(NDIMS == 1)
  {
    std::array<T, NDIMS> coord;
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t idim = 0; idim < NDIMS; ++idim) {
        coord[idim] = node_coords(elem, inode, idim);
      }

      auto init_value = initial.get_initial(coord);

      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u(elem, inode, ivar) = init_value[ivar];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int& ielem,
                                         const int& jelem) const
    requires(NDIMS == 2)
  {
    std::array<T, NDIMS> coord;
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t idim = 0; idim < NDIMS; ++idim) {
        coord[idim] = node_coords(ielem, jelem, inode, idim);
      }

      auto init_value = initial.get_initial(coord);

      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u(ielem, jelem, inode, ivar) = init_value[ivar];
      }
    }
  }

  std::array<std::size_t, NDIMS> n_elems;
  std::size_t n_dofs;
  std::size_t NVARS;
  DataArray u;
  CoordArray node_coords;
  const Functor initial;
};
} // namespace DGSEM
