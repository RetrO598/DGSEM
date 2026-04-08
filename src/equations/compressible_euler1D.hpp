#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <equations/equations_base.hpp>

namespace DGSEM {
namespace equations {
template <class T>
class CompressibleEuler1D : public Equations1DBase {
public:
  using value_type = T;
  CompressibleEuler1D(const T& gamma) : gamma_(gamma) {}

  constexpr static std::size_t NDIMS = 1;
  constexpr static std::size_t NVARS = 3;

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS>& u,
                                     std::size_t dim) const {
    value_type rho = u[0];
    value_type mom = u[1];
    value_type rhoE = u[2];
    value_type u_vel = mom / rho;
    value_type gamma = 1.4;
    value_type p = (gamma - 1.0) * (rhoE - 0.5 * rho * u_vel * u_vel);
    return {mom, mom * u_vel + p, u_vel * (rhoE + p)};
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_wave_speed(const std::array<value_type, NVARS>& u_ll,
                            const std::array<value_type, NVARS>& u_rr) const {
    value_type rho_ll = u_ll[0];
    value_type mom_ll = u_ll[1];
    value_type rhoE_ll = u_ll[2];
    value_type rho_rr = u_rr[0];
    value_type mom_rr = u_rr[1];
    value_type rhoE_rr = u_rr[2];
    value_type u_vel_ll = mom_ll / rho_ll;
    value_type u_vel_rr = mom_rr / rho_rr;
    value_type gamma = gamma_;
    value_type p_ll =
        (gamma - 1.0) * (rhoE_ll - 0.5 * rho_ll * u_vel_ll * u_vel_ll);
    value_type p_rr =
        (gamma - 1.0) * (rhoE_rr - 0.5 * rho_rr * u_vel_rr * u_vel_rr);
    value_type a_ll = std::sqrt(gamma * p_ll / rho_ll);
    value_type a_rr = std::sqrt(gamma * p_rr / rho_rr);
    return std::max(std::abs(u_vel_ll), std::abs(u_vel_rr)) +
           std::max(a_ll, a_rr);
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_gamma() const { return gamma_; }

private:
  value_type gamma_ = 1.4;
};
} // namespace equations
} // namespace DGSEM
