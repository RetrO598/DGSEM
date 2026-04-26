#pragma once

#include <array>
#include <cstddef>
#include <equations/compressible_euler3D.hpp>
#include <equations/equations_base.hpp>

namespace DGSEM {
namespace equations {

template <class T>
class CompressibleNavierStokes3D : public Equations3DBase {
public:
  using value_type = T;

  KOKKOS_INLINE_FUNCTION
  explicit CompressibleNavierStokes3D(
      const T& gamma = static_cast<T>(1.4),
      const T& dynamic_viscosity = static_cast<T>(6.25e-4),
      const T& prandtl_number = static_cast<T>(0.72))
      : gamma_(gamma), mu_(dynamic_viscosity), prandtl_(prandtl_number),
        kappa_(gamma / ((gamma - static_cast<T>(1.0)) * prandtl_number)) {}

  constexpr static std::size_t NDIMS = 3;
  constexpr static std::size_t NVARS = 5;
  constexpr static std::size_t NGRAD_VARS = 5;

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS>& u,
                                     std::size_t dim) const {
    return euler_.flux(u, dim);
  }

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS>
  flux(const std::array<value_type, NVARS>& u,
       const std::array<value_type, NDIMS>& normal) const {
    return euler_.flux(u, normal);
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_wave_speed(const std::array<value_type, NVARS>& u_ll,
                            const std::array<value_type, NVARS>& u_rr,
                            std::size_t dim) const {
    return euler_.get_wave_speed(u_ll, u_rr, dim);
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_wave_speed(const std::array<value_type, NVARS>& u_ll,
                            const std::array<value_type, NVARS>& u_rr,
                            const std::array<value_type, NDIMS>& normal) const {
    return euler_.get_wave_speed(u_ll, u_rr, normal);
  }

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NGRAD_VARS>
  gradient_variables(const std::array<value_type, NVARS>& u) const {
    const value_type rho = u[0];
    const value_type inv_rho = static_cast<value_type>(1.0) / rho;
    const value_type v1 = u[1] * inv_rho;
    const value_type v2 = u[2] * inv_rho;
    const value_type v3 = u[3] * inv_rho;
    const value_type p =
        (gamma_ - static_cast<value_type>(1.0)) *
        (u[4] - static_cast<value_type>(0.5) *
                    (u[1] * u[1] + u[2] * u[2] + u[3] * u[3]) * inv_rho);
    const value_type temperature = p * inv_rho;

    return {rho, v1, v2, v3, temperature};
  }

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS> viscous_flux(
      const std::array<value_type, NGRAD_VARS>& q,
      const std::array<std::array<value_type, NGRAD_VARS>, NDIMS>& grad_q,
      std::size_t dim) const {
    const value_type v1 = q[1];
    const value_type v2 = q[2];
    const value_type v3 = q[3];

    const value_type dv1dx = grad_q[0][1];
    const value_type dv2dx = grad_q[0][2];
    const value_type dv3dx = grad_q[0][3];
    const value_type dTdx = grad_q[0][4];

    const value_type dv1dy = grad_q[1][1];
    const value_type dv2dy = grad_q[1][2];
    const value_type dv3dy = grad_q[1][3];
    const value_type dTdy = grad_q[1][4];

    const value_type dv1dz = grad_q[2][1];
    const value_type dv2dz = grad_q[2][2];
    const value_type dv3dz = grad_q[2][3];
    const value_type dTdz = grad_q[2][4];

    const value_type tau11 = (static_cast<value_type>(4.0) * dv1dx -
                              static_cast<value_type>(2.0) * (dv2dy + dv3dz)) /
                             static_cast<value_type>(3.0);
    const value_type tau22 = (static_cast<value_type>(4.0) * dv2dy -
                              static_cast<value_type>(2.0) * (dv1dx + dv3dz)) /
                             static_cast<value_type>(3.0);
    const value_type tau33 = (static_cast<value_type>(4.0) * dv3dz -
                              static_cast<value_type>(2.0) * (dv1dx + dv2dy)) /
                             static_cast<value_type>(3.0);
    const value_type tau12 = dv1dy + dv2dx;
    const value_type tau13 = dv1dz + dv3dx;
    const value_type tau23 = dv2dz + dv3dy;

    const value_type q1 = kappa_ * dTdx;
    const value_type q2 = kappa_ * dTdy;
    const value_type q3 = kappa_ * dTdz;

    if (dim == 0) {
      return {static_cast<value_type>(0.0), mu_ * tau11, mu_ * tau12,
              mu_ * tau13, mu_ * (v1 * tau11 + v2 * tau12 + v3 * tau13 + q1)};
    } else if (dim == 1) {
      return {static_cast<value_type>(0.0), mu_ * tau12, mu_ * tau22,
              mu_ * tau23, mu_ * (v1 * tau12 + v2 * tau22 + v3 * tau23 + q2)};
    }

    return {static_cast<value_type>(0.0), mu_ * tau13, mu_ * tau23, mu_ * tau33,
            mu_ * (v1 * tau13 + v2 * tau23 + v3 * tau33 + q3)};
  }

  KOKKOS_INLINE_FUNCTION
  value_type get_gamma() const { return gamma_; }

  KOKKOS_INLINE_FUNCTION
  value_type get_mu() const { return mu_; }

  KOKKOS_INLINE_FUNCTION
  value_type get_prandtl() const { return prandtl_; }

private:
  value_type gamma_ = static_cast<value_type>(1.4);
  value_type mu_ = static_cast<value_type>(6.25e-4);
  value_type prandtl_ = static_cast<value_type>(0.72);
  value_type kappa_ = static_cast<value_type>(0.0);
  CompressibleEuler3D<value_type> euler_{gamma_};
};

} // namespace equations
} // namespace DGSEM
