#pragma once

#include <Kokkos_Macros.hpp>
#include <cstddef>

namespace DGSEM {
namespace utils {

template <std::size_t NNodes>
KOKKOS_INLINE_FUNCTION constexpr std::size_t local_dof(std::size_t inode,
                                                       std::size_t jnode) {
  return jnode * NNodes + inode;
}

} // namespace utils
} // namespace DGSEM
