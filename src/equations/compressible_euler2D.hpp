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
class CompressibleEuler2D : public Equations2DBase {
public:
  using value_type = T;

  explicit CompressibleEuler2D(const T& gamma = static_cast<T>(1.4))
      : gamma_(gamma) {}

  constexpr static std::size_t NDIMS = 2;
  constexpr static std::size_t NVARS = 4;

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS>& u,
                                     std::size_t dim) const {
    const value_type rho = u[0];
    const value_type rhou = u[1];
    const value_type rhov = u[2];
    const value_type rhoE = u[3];

    const value_type u_vel = rhou / rho;
    const value_type v_vel = rhov / rho;
    const value_type kinetic =
        static_cast<value_type>(0.5) * rho * (u_vel * u_vel + v_vel * v_vel);
    const value_type p =
        (gamma_ - static_cast<value_type>(1.0)) * (rhoE - kinetic);

    if (dim == 0) {
      return {rhou, rhou * u_vel + p, rhou * v_vel, u_vel * (rhoE + p)};
    }

    return {rhov, rhov * u_vel, rhov * v_vel + p, v_vel * (rhoE + p)};
  }

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS>
  flux(const std::array<value_type, NVARS>& u,
       const std::array<value_type, NDIMS>& normal) const {
    const auto flux_x = flux(u, 0);
    const auto flux_y = flux(u, 1);
    std::array<value_type, NVARS> normal_flux{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      normal_flux[var] = normal[0] * flux_x[var] + normal[1] * flux_y[var];
    }
    return normal_flux;
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_wave_speed(const std::array<value_type, NVARS>& u_ll,
                            const std::array<value_type, NVARS>& u_rr,
                            std::size_t dim) const {
    const value_type rho_ll = u_ll[0];
    const value_type rhou_ll = u_ll[1];
    const value_type rhov_ll = u_ll[2];
    const value_type rhoE_ll = u_ll[3];

    const value_type rho_rr = u_rr[0];
    const value_type rhou_rr = u_rr[1];
    const value_type rhov_rr = u_rr[2];
    const value_type rhoE_rr = u_rr[3];

    const value_type u_vel_ll = rhou_ll / rho_ll;
    const value_type v_vel_ll = rhov_ll / rho_ll;
    const value_type u_vel_rr = rhou_rr / rho_rr;
    const value_type v_vel_rr = rhov_rr / rho_rr;

    const value_type p_ll =
        (gamma_ - static_cast<value_type>(1.0)) *
        (rhoE_ll - static_cast<value_type>(0.5) * rho_ll *
                       (u_vel_ll * u_vel_ll + v_vel_ll * v_vel_ll));
    const value_type p_rr =
        (gamma_ - static_cast<value_type>(1.0)) *
        (rhoE_rr - static_cast<value_type>(0.5) * rho_rr *
                       (u_vel_rr * u_vel_rr + v_vel_rr * v_vel_rr));

    const value_type a_ll = std::sqrt(gamma_ * p_ll / rho_ll);
    const value_type a_rr = std::sqrt(gamma_ * p_rr / rho_rr);

    const value_type normal_ll = (dim == 0) ? u_vel_ll : v_vel_ll;
    const value_type normal_rr = (dim == 0) ? u_vel_rr : v_vel_rr;

    return std::max(std::abs(normal_ll) + a_ll, std::abs(normal_rr) + a_rr);
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_wave_speed(const std::array<value_type, NVARS>& u_ll,
                            const std::array<value_type, NVARS>& u_rr,
                            const std::array<value_type, NDIMS>& normal) const {
    const value_type rho_ll = u_ll[0];
    const value_type rhou_ll = u_ll[1];
    const value_type rhov_ll = u_ll[2];
    const value_type rhoE_ll = u_ll[3];

    const value_type rho_rr = u_rr[0];
    const value_type rhou_rr = u_rr[1];
    const value_type rhov_rr = u_rr[2];
    const value_type rhoE_rr = u_rr[3];

    const value_type u_vel_ll = rhou_ll / rho_ll;
    const value_type v_vel_ll = rhov_ll / rho_ll;
    const value_type u_vel_rr = rhou_rr / rho_rr;
    const value_type v_vel_rr = rhov_rr / rho_rr;

    const value_type p_ll =
        (gamma_ - static_cast<value_type>(1.0)) *
        (rhoE_ll - static_cast<value_type>(0.5) * rho_ll *
                       (u_vel_ll * u_vel_ll + v_vel_ll * v_vel_ll));
    const value_type p_rr =
        (gamma_ - static_cast<value_type>(1.0)) *
        (rhoE_rr - static_cast<value_type>(0.5) * rho_rr *
                       (u_vel_rr * u_vel_rr + v_vel_rr * v_vel_rr));

    const value_type a_ll = std::sqrt(gamma_ * p_ll / rho_ll);
    const value_type a_rr = std::sqrt(gamma_ * p_rr / rho_rr);
    const value_type normal_norm =
        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
    const value_type normal_vel_ll =
        u_vel_ll * normal[0] + v_vel_ll * normal[1];
    const value_type normal_vel_rr =
        u_vel_rr * normal[0] + v_vel_rr * normal[1];

    return std::max(std::abs(normal_vel_ll), std::abs(normal_vel_rr)) +
           std::max(a_ll, a_rr) * normal_norm;
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_gamma() const { return gamma_; }

private:
  value_type gamma_ = static_cast<value_type>(1.4);
};

} // namespace equations
} // namespace DGSEM
