#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>

namespace DGSEM::utils {

template <std::size_t NDIMS, class Functor>
void parallel_for_cells(const char* label,
                        const std::array<std::size_t, NDIMS>& n_cells,
                        const Functor& functor) {
  static_assert(NDIMS >= 1 && NDIMS <= 3,
                "parallel_for_cells currently supports 1D, 2D, and 3D grids.");

  if constexpr (NDIMS == 1) {
    Kokkos::parallel_for(label, n_cells[0], functor);
  } else if constexpr (NDIMS == 2) {
    Kokkos::parallel_for(label,
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_cells[0], n_cells[1]}),
                         functor);
  } else {
    Kokkos::parallel_for(label,
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {n_cells[0], n_cells[1], n_cells[2]}),
                         functor);
  }
}

} // namespace DGSEM::utils
