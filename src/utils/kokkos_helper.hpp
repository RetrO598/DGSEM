#pragma once

#include "Kokkos_Core.hpp"
#include <base/base.hpp>

namespace DGSEM {
namespace utils {
template <class ViewType>
void clone_view_shape(ViewType& dst, const ViewType& src) {
  if constexpr (ViewType::rank == 1) {
    Kokkos::realloc(dst, src.extent(0));
  } else if constexpr (ViewType::rank == 2) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1));
  } else if constexpr (ViewType::rank == 3) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2));
  } else if constexpr (ViewType::rank == 4) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2),
                    src.extent(3));
  } else if constexpr (ViewType::rank == 5) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2),
                    src.extent(3), src.extent(4));
  } else if constexpr (ViewType::rank == 6) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2),
                    src.extent(3), src.extent(4), src.extent(5));
  } else {
    static_assert(ViewType::rank <= 6, "Unsupported rank");
  }

}
} // namespace utils
} // namespace DGSEM
