#pragma once

#include <Kokkos_Macros.hpp>
#include <array>
#include <cstddef>

namespace DGSEM::detail {

template <class T, std::size_t NVARS>
KOKKOS_INLINE_FUNCTION constexpr std::array<T, NVARS>
lax_friedrichs_flux_from_physical_fluxes(
    const std::array<T, NVARS>& flux_L, const std::array<T, NVARS>& flux_R,
    const std::array<T, NVARS>& uL, const std::array<T, NVARS>& uR,
    T max_speed) {
  std::array<T, NVARS> flux{};
  for (std::size_t i = 0; i < NVARS; ++i) {
    flux[i] = static_cast<T>(0.5) * (flux_L[i] + flux_R[i]) -
              static_cast<T>(0.5) * max_speed * (uR[i] - uL[i]);
  }
  return flux;
}

template <class T, std::size_t NVARS>
KOKKOS_INLINE_FUNCTION constexpr std::array<T, NVARS>
central_flux_from_physical_fluxes(const std::array<T, NVARS>& flux_L,
                                  const std::array<T, NVARS>& flux_R) {
  std::array<T, NVARS> flux{};
  for (std::size_t i = 0; i < NVARS; ++i) {
    flux[i] = static_cast<T>(0.5) * (flux_L[i] + flux_R[i]);
  }
  return flux;
}

} // namespace DGSEM::detail
