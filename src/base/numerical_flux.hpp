#pragma once

#include <Kokkos_Macros.hpp>
#include <algorithm>
#include <array>
#include <base/math_utils.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <equations/equations.hpp>

namespace DGSEM {
template <class Equations>
struct LaxFriedrichsFlux {};

template <class T>
struct LaxFriedrichsFlux<equations::LinearScalarAdvection1D<T>> {
  using traits =
      equations::EquationTraits<equations::LinearScalarAdvection1D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::LinearScalarAdvection1D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto speed = eq.get_wave_speed();
    auto max_speed = std::abs(speed);
    std::array<T, NVARS> flux_L = eq.flux(uL, 0);
    std::array<T, NVARS> flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }
};

template <class T>
struct LaxFriedrichsFlux<equations::BuckleyLeverett1D<T>> {
  using traits = equations::EquationTraits<equations::BuckleyLeverett1D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::BuckleyLeverett1D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto speed = eq.get_wave_speed();
    auto max_speed = std::abs(speed);
    std::array<T, NVARS> flux_L = eq.flux(uL, 0);
    std::array<T, NVARS> flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }
};

template <class T>
struct LaxFriedrichsFlux<equations::InviscidBurgers1D<T>> {
  using traits = equations::EquationTraits<equations::InviscidBurgers1D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::InviscidBurgers1D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto max_speed = eq.get_wave_speed(uL, uR);
    std::array<T, NVARS> flux_L = eq.flux(uL, 0);
    std::array<T, NVARS> flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }
};

template <class T>
struct LaxFriedrichsFlux<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::CompressibleEuler1D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto max_speed = eq.get_wave_speed(uL, uR);
    std::array<T, NVARS> flux_L = eq.flux(uL, 0);
    std::array<T, NVARS> flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }

    return flux;
  }
};

template <class T>
struct LaxFriedrichsFlux<equations::CompressibleEuler2D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler2D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::CompressibleEuler2D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto max_speed = eq.get_wave_speed(uL, uR, dim);
    std::array<T, NVARS> flux_L = eq.flux(uL, dim);
    std::array<T, NVARS> flux_R = eq.flux(uR, dim);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::CompressibleEuler2D<T>& eq,
                 const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR,
                 const std::array<value_type, NDIMS>& normal) {
    auto max_speed = eq.get_wave_speed(uL, uR, normal);
    std::array<T, NVARS> flux_L = eq.flux(uL, normal);
    std::array<T, NVARS> flux_R = eq.flux(uR, normal);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]) -
                static_cast<value_type>(0.5) * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }
};

template <class Equations>
struct CentralFlux {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const Equations& eq, const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR, std::size_t dim = 0) {
    auto flux_L = eq.flux(uL, dim);
    auto flux_R = eq.flux(uR, dim);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]);
    }
    return flux;
  }

  KOKKOS_INLINE_FUNCTION constexpr static std::array<value_type, NVARS>
  numerical_flux(const Equations& eq, const std::array<value_type, NVARS>& uL,
                 const std::array<value_type, NVARS>& uR,
                 const std::array<value_type, NDIMS>& normal)
    requires(NDIMS == 2)
  {
    auto flux_L = eq.flux(uL, normal);
    auto flux_R = eq.flux(uR, normal);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = static_cast<value_type>(0.5) * (flux_L[i] + flux_R[i]);
    }
    return flux;
  }
};

template <class Equations>
struct flux_godunov {};

template <class T>
struct flux_godunov<equations::LinearScalarAdvection1D<T>> {
  using traits =
      equations::EquationTraits<equations::LinearScalarAdvection1D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<T, NVARS>
  numerical_flux(const equations::LinearScalarAdvection1D<T>& eq,
                 const std::array<T, NVARS>& u_ll,
                 const std::array<T, NVARS>& u_rr) {
    auto speed = eq.get_wave_speed();
    if (speed >= 0) {
      return eq.flux(u_ll, 0);
    } else {
      return eq.flux(u_rr, 0);
    }
  }
};

template <class Equations>
struct HLLCFlux;

template <class T>
struct HLLCFlux<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<T, NVARS>
  numerical_flux(const equations::CompressibleEuler1D<T>& eq,
                 const std::array<T, NVARS>& u_ll,
                 const std::array<T, NVARS>& u_rr) {
    T rho_ll = u_ll[0];
    T mom_ll = u_ll[1];
    T rhoE_ll = u_ll[2];
    T rho_rr = u_rr[0];
    T mom_rr = u_rr[1];
    T rhoE_rr = u_rr[2];
    T v1_ll = mom_ll / rho_ll;
    T v1_rr = mom_rr / rho_rr;

    T E_ll = rhoE_ll / rho_ll;
    T E_rr = rhoE_rr / rho_rr;

    T gamma = eq.get_gamma();

    T p_ll = (gamma - static_cast<T>(1.0)) *
             (rhoE_ll - static_cast<T>(0.5) * rho_ll * v1_ll * v1_ll);
    T p_rr = (gamma - static_cast<T>(1.0)) *
             (rhoE_rr - static_cast<T>(0.5) * rho_rr * v1_rr * v1_rr);

    T c_ll = std::sqrt(gamma * p_ll / rho_ll);
    T c_rr = std::sqrt(gamma * p_rr / rho_rr);

    auto flux_ll = eq.flux(u_ll, 0);
    auto flux_rr = eq.flux(u_rr, 0);

    T sqrt_rho_ll = std::sqrt(rho_ll);
    T sqrt_rho_rr = std::sqrt(rho_rr);
    T sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr;
    T vel_L = v1_ll;
    T vel_R = v1_rr;

    T vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho;
    T ekin_roe = static_cast<T>(0.5) * vel_roe * vel_roe;
    T H_ll = (rhoE_ll + p_ll) / rho_ll;
    T H_rr = (rhoE_rr + p_rr) / rho_rr;
    T H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho;
    T c_roe = std::sqrt((gamma - static_cast<T>(1.0)) * (H_roe - ekin_roe));

    T Ssl = std::min(vel_L - c_ll, vel_roe - c_roe);
    T Ssr = std::max(vel_R + c_rr, vel_roe + c_roe);

    T sMu_L = Ssl - vel_L;
    T sMu_R = Ssr - vel_R;

    T f1, f2, f3;
    if (Ssl >= 0) {
      f1 = flux_ll[0];
      f2 = flux_ll[1];
      f3 = flux_ll[2];
    } else if (Ssr <= 0) {
      f1 = flux_rr[0];
      f2 = flux_rr[1];
      f3 = flux_rr[2];
    } else {
      T SStar =
          (p_rr - p_ll + rho_ll * vel_L * sMu_L - rho_rr * vel_R * sMu_R) /
          (rho_ll * sMu_L - rho_rr * sMu_R);

      if (Ssl <= 0 && SStar >= 0) {
        T densStar = rho_ll * sMu_L / (Ssl - SStar);
        T enerStar = E_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L));
        T UStar1 = densStar;
        T Ustar2 = densStar * SStar;
        T Ustar3 = densStar * enerStar;

        f1 = flux_ll[0] + Ssl * (UStar1 - rho_ll);
        f2 = flux_ll[1] + Ssl * (Ustar2 - mom_ll);
        f3 = flux_ll[2] + Ssl * (Ustar3 - rhoE_ll);
      } else {
        T densStar = rho_rr * sMu_R / (Ssr - SStar);
        T enerStar = E_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R));
        T UStar1 = densStar;
        T Ustar2 = densStar * SStar;
        T Ustar3 = densStar * enerStar;

        f1 = flux_rr[0] + Ssr * (UStar1 - rho_rr);
        f2 = flux_rr[1] + Ssr * (Ustar2 - mom_rr);
        f3 = flux_rr[2] + Ssr * (Ustar3 - rhoE_rr);
      }
    }
    return {f1, f2, f3};
  }
};

template <class Equations>
struct ChandrashekarFlux;

template <class T>
struct ChandrashekarFlux<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<T, NVARS>
  numerical_flux(const equations::CompressibleEuler1D<T>& eq,
                 const std::array<T, NVARS>& u_ll,
                 const std::array<T, NVARS>& u_rr) {
    T rho_ll = u_ll[0];
    T mom_ll = u_ll[1];
    T rhoE_ll = u_ll[2];
    T rho_rr = u_rr[0];
    T mom_rr = u_rr[1];
    T rhoE_rr = u_rr[2];
    T v1_ll = mom_ll / rho_ll;
    T v1_rr = mom_rr / rho_rr;

    T E_ll = rhoE_ll / rho_ll;
    T E_rr = rhoE_rr / rho_rr;

    T gamma = eq.get_gamma();

    T p_ll = (gamma - static_cast<T>(1.0)) *
             (rhoE_ll - static_cast<T>(0.5) * rho_ll * v1_ll * v1_ll);
    T p_rr = (gamma - static_cast<T>(1.0)) *
             (rhoE_rr - static_cast<T>(0.5) * rho_rr * v1_rr * v1_rr);

    T beta_ll = static_cast<T>(0.5) * rho_ll / p_ll;
    T beta_rr = static_cast<T>(0.5) * rho_rr / p_rr;
    T specific_kin_ll = static_cast<T>(0.5) * v1_ll * v1_ll;
    T specific_kin_rr = static_cast<T>(0.5) * v1_rr * v1_rr;

    T rho_avg = static_cast<T>(0.5) * (rho_ll + rho_rr);
    T rho_mean = ln_mean(rho_ll, rho_rr);
    T beta_mean = ln_mean(beta_ll, beta_rr);
    T beta_avg = static_cast<T>(0.5) * (beta_ll + beta_rr);
    T v1_avg = static_cast<T>(0.5) * (v1_ll + v1_rr);
    T p_mean = static_cast<T>(0.5) * rho_avg / beta_avg;
    T velocity_square_avg = specific_kin_ll + specific_kin_rr;

    T f1 = rho_mean * v1_avg;
    T f2 = f1 * v1_avg + p_mean;
    T f3 = f1 * static_cast<T>(0.5) *
                   (static_cast<T>(1.0) / (gamma - static_cast<T>(1.0)) /
                        beta_mean -
                    velocity_square_avg) +
           f2 * v1_avg;
    return {f1, f2, f3};
  }
};

template <class T>
struct ChandrashekarFlux<equations::CompressibleEuler2D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler2D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<T, NVARS>
  numerical_flux(const equations::CompressibleEuler2D<T>& eq,
                 const std::array<T, NVARS>& u_ll,
                 const std::array<T, NVARS>& u_rr,
                 const std::array<T, NDIMS>& normal) {
    const T rho_ll = u_ll[0];
    const T rhou_ll = u_ll[1];
    const T rhov_ll = u_ll[2];
    const T rhoE_ll = u_ll[3];

    const T rho_rr = u_rr[0];
    const T rhou_rr = u_rr[1];
    const T rhov_rr = u_rr[2];
    const T rhoE_rr = u_rr[3];

    const T v1_ll = rhou_ll / rho_ll;
    const T v2_ll = rhov_ll / rho_ll;
    const T v1_rr = rhou_rr / rho_rr;
    const T v2_rr = rhov_rr / rho_rr;

    const T v_dot_n_ll = v1_ll * normal[0] + v2_ll * normal[1];
    const T v_dot_n_rr = v1_rr * normal[0] + v2_rr * normal[1];

    const T specific_kin_ll =
        static_cast<T>(0.5) * (v1_ll * v1_ll + v2_ll * v2_ll);
    const T specific_kin_rr =
        static_cast<T>(0.5) * (v1_rr * v1_rr + v2_rr * v2_rr);

    const T gamma = eq.get_gamma();
    const T p_ll = (gamma - static_cast<T>(1.0)) *
                   (rhoE_ll - static_cast<T>(0.5) *
                                  rho_ll * (v1_ll * v1_ll + v2_ll * v2_ll));
    const T p_rr = (gamma - static_cast<T>(1.0)) *
                   (rhoE_rr - static_cast<T>(0.5) *
                                  rho_rr * (v1_rr * v1_rr + v2_rr * v2_rr));

    const T beta_ll = static_cast<T>(0.5) * rho_ll / p_ll;
    const T beta_rr = static_cast<T>(0.5) * rho_rr / p_rr;
    const T rho_avg = static_cast<T>(0.5) * (rho_ll + rho_rr);
    const T rho_mean = ln_mean(rho_ll, rho_rr);
    const T beta_mean = ln_mean(beta_ll, beta_rr);
    const T beta_avg = static_cast<T>(0.5) * (beta_ll + beta_rr);
    const T v1_avg = static_cast<T>(0.5) * (v1_ll + v1_rr);
    const T v2_avg = static_cast<T>(0.5) * (v2_ll + v2_rr);
    const T p_mean = static_cast<T>(0.5) * rho_avg / beta_avg;
    const T velocity_square_avg = specific_kin_ll + specific_kin_rr;

    const T f1 = rho_mean * static_cast<T>(0.5) * (v_dot_n_ll + v_dot_n_rr);
    const T f2 = f1 * v1_avg + p_mean * normal[0];
    const T f3 = f1 * v2_avg + p_mean * normal[1];
    const T f4 =
        f1 * static_cast<T>(0.5) *
            (static_cast<T>(1.0) / (gamma - static_cast<T>(1.0)) / beta_mean -
             velocity_square_avg) +
        f2 * v1_avg + f3 * v2_avg;

    return {f1, f2, f3, f4};
  }
};

template <class Equations>
struct ChandrashekarESFlux;

template <class T>
struct ChandrashekarESFlux<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION constexpr static std::array<T, NVARS>
  numerical_flux(const equations::CompressibleEuler1D<T>& eq,
                 const std::array<T, NVARS>& u_ll,
                 const std::array<T, NVARS>& u_rr) {
    T rho_ll = u_ll[0];
    T mom_ll = u_ll[1];
    T rhoE_ll = u_ll[2];
    T rho_rr = u_rr[0];
    T mom_rr = u_rr[1];
    T rhoE_rr = u_rr[2];
    T v1_ll = mom_ll / rho_ll;
    T v1_rr = mom_rr / rho_rr;

    T E_ll = rhoE_ll / rho_ll;
    T E_rr = rhoE_rr / rho_rr;

    T gamma = eq.get_gamma();

    T p_ll = (gamma - static_cast<T>(1.0)) *
             (rhoE_ll - static_cast<T>(0.5) * rho_ll * v1_ll * v1_ll);
    T p_rr = (gamma - static_cast<T>(1.0)) *
             (rhoE_rr - static_cast<T>(0.5) * rho_rr * v1_rr * v1_rr);

    T beta_ll = static_cast<T>(0.5) * rho_ll / p_ll;
    T beta_rr = static_cast<T>(0.5) * rho_rr / p_rr;
    T specific_kin_ll = static_cast<T>(0.5) * v1_ll * v1_ll;
    T specific_kin_rr = static_cast<T>(0.5) * v1_rr * v1_rr;

    T rho_avg = static_cast<T>(0.5) * (rho_ll + rho_rr);
    T rho_mean = ln_mean(rho_ll, rho_rr);
    T beta_mean = ln_mean(beta_ll, beta_rr);
    T beta_avg = static_cast<T>(0.5) * (beta_ll + beta_rr);
    T v1_avg = static_cast<T>(0.5) * (v1_ll + v1_rr);
    T p_mean = static_cast<T>(0.5) * rho_avg / beta_avg;
    T velocity_square_avg = specific_kin_ll + specific_kin_rr;

    T f1 = rho_mean * v1_avg;
    T f2 = f1 * v1_avg + p_mean;
    T f3 = f1 * static_cast<T>(0.5) *
                   (static_cast<T>(1.0) / (gamma - static_cast<T>(1.0)) /
                        beta_mean -
                    velocity_square_avg) +
           f2 * v1_avg;

    auto max_speed = eq.get_wave_speed(u_ll, u_rr);
    // T max_speed = std::abs(v1_avg) + std::sqrt(gamma / (2.0 * beta_mean));

    f1 = f1 - static_cast<T>(0.5) * max_speed * (rho_rr - rho_ll);
    f2 = f2 - static_cast<T>(0.5) * max_speed * (mom_rr - mom_ll);
    f3 =
        f3 -
        static_cast<T>(0.5) * max_speed *
            ((static_cast<T>(0.5) *
                  (static_cast<T>(1.0) /
                   (gamma - static_cast<T>(1.0)) / beta_mean) +
              static_cast<T>(0.5) * v1_ll * v1_rr) *
                 (rho_rr - rho_ll) +
             rho_avg * v1_avg * (v1_rr - v1_ll) +
             rho_avg / (static_cast<T>(2.0) *
                        (gamma - static_cast<T>(1.0))) *
                 (static_cast<T>(1.0) / beta_rr -
                  static_cast<T>(1.0) / beta_ll));

    return {f1, f2, f3};
  }
};
} // namespace DGSEM
