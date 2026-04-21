#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/utils.hpp>

namespace DGSEM {
template <class Equations>
struct VolumeAverageEuler {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;

  static_assert(traits::NVARS == traits::NDIMS + 2,
                "VolumeAverageEuler only supports Euler-like systems.");

  Equations eq;
  T total_volume;
  T kinetic_energy_integral;
  T entropy_integral;

  KOKKOS_INLINE_FUNCTION
  void init() {
    total_volume = T{0};
    kinetic_energy_integral = T{0};
    entropy_integral = T{0};
    eq = Equations();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::Array<T, traits::NVARS>& u,
                  const T volume_weight) {
    const T rho = u[0];

    T momentum_sq = T{0};
    for (std::size_t dim = 0; dim < traits::NDIMS; ++dim) {
      momentum_sq += u[dim + 1] * u[dim + 1];
    }

    const T kinetic_energy_density = static_cast<T>(0.5) * momentum_sq / rho;

    auto prim = DGSEM::utils::cons_to_prim(u, eq.get_gamma());
    const T pressure = prim[traits::NDIMS + 1];

    const T specific_entropy =
        Kokkos::log(pressure) - eq.get_gamma() * Kokkos::log(rho);
    const T entropy_density =
        -rho * specific_entropy / (eq.get_gamma() - static_cast<T>(1.0));

    total_volume += volume_weight;
    kinetic_energy_integral += volume_weight * kinetic_energy_density;
    entropy_integral += volume_weight * entropy_density;
  }

  KOKKOS_INLINE_FUNCTION
  void join(const VolumeAverageEuler& other) {
    total_volume += other.total_volume;
    kinetic_energy_integral += other.kinetic_energy_integral;
    entropy_integral += other.entropy_integral;
  }

  KOKKOS_INLINE_FUNCTION
  T mean_kinetic_energy() const {
    return kinetic_energy_integral / total_volume;
  }

  KOKKOS_INLINE_FUNCTION
  T mean_entropy() const { return entropy_integral / total_volume; }
};
} // namespace DGSEM
