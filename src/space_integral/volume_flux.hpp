#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <equations/equations.hpp>

#include <vector>
#include <xtensor/core/xtensor_forward.hpp>
#include "base/global_def.hpp"

namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct VolumeIntegralWeak;

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 1> {

  template <class Basis, class Equations>
  inline constexpr static void
  weak_form_kernel(std::size_t ielem, const Equations &eq,
                   const xt::xarray<T> &u, xt::xarray<T> &du) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      std::array<T, NVARS> flux;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }
      flux = eq.flux(u_node, 0);

      for (std::size_t iinode = 0; iinode < Basis::NNodes; ++iinode) {
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, iinode, var) =
              du(ielem, iinode, var) +
              Basis::derivative_dhat(iinode, inode) * flux[var];
        }
      }
    }
  }

  template <class Basis, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  weak_form_kernel_kokkos(std::size_t ielem, const Equations &eq,
                          const ArrayU &u, ArrayDu &du) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      std::array<T, NVARS> flux;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }
      flux = eq.flux(u_node, 0);

      for (std::size_t iinode = 0; iinode < Basis::NNodes; ++iinode) {
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, iinode, var) =
              du(ielem, iinode, var) +
              Basis::derivative_dhat(iinode, inode) * flux[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct VolumeIntegralSplit;

template <class T, std::size_t NVARS, class NumericFlux>
struct VolumeIntegralSplit<T, NVARS, NumericFlux, 1> {

  template <class Basis, class Equations>
  inline constexpr static void
  split_form_kernel(std::size_t ielem, const Equations &eq,
                    const xt::xarray<T> &u, xt::xarray<T> &du) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < Basis::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              Basis::derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              Basis::derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }

  template <class Basis, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  split_form_kernel_kokkos(std::size_t ielem, const Equations &eq,
                           const ArrayU &u, ArrayDu &du) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < Basis::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              Basis::derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              Basis::derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }

  template <class Basis, class Equations>
  inline constexpr static void
  split_form_kernel(std::size_t ielem, const Equations &eq,
                    const xt::xarray<T> &u, xt::xarray<T> &du, T alpha) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < Basis::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              alpha * Basis::derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              alpha * Basis::derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }

  template <class Basis, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  split_form_kernel_kokkos(std::size_t ielem, const Equations &eq,
                           const ArrayU &u, ArrayDu &du, T alpha) {
    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < Basis::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              alpha * Basis::derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              alpha * Basis::derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct FinitVolumeIntegral;

template <class T, std::size_t NVARS, class NumericFlux>
struct FinitVolumeIntegral<T, NVARS, NumericFlux, 1> {

  template <class Basis, class Equations>
  inline constexpr static void fv_kernel(std::size_t ielem, const Equations &eq,
                                         const xt::xarray<T> &u,
                                         xt::xarray<T> &du, T alpha) {
    std::array<std::array<T, NVARS>, Basis::NNodes + 1> fstar1_L{};
    std::array<std::array<T, NVARS>, Basis::NNodes + 1> fstar1_R{};
    for (std::size_t inode = 1; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_ll{};
      std::array<T, NVARS> u_rr{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_ll[var] = u(ielem, inode - 1, var);
        u_rr[var] = u(ielem, inode, var);
      }

      auto fstar1 = NumericFlux::numerical_flux(eq, u_ll, u_rr);
      for (std::size_t var = 0; var < NVARS; ++var) {
        fstar1_L[inode][var] = fstar1[var];
        fstar1_R[inode][var] = fstar1[var];
      }
    }

    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, inode, var) =
            du(ielem, inode, var) +
            (alpha * (Basis::inv_weights[inode] *
                      (fstar1_L[inode + 1][var] - fstar1_R[inode][var])));
      }
    }
  }

  template <class Basis, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  fv_kernel_kokkos(std::size_t ielem, const Equations &eq, const ArrayU &u,
                   ArrayDu &du, T alpha) {
    std::array<std::array<T, NVARS>, Basis::NNodes + 1> fstar1_L{};
    std::array<std::array<T, NVARS>, Basis::NNodes + 1> fstar1_R{};
    for (std::size_t inode = 1; inode < Basis::NNodes; ++inode) {
      std::array<T, NVARS> u_ll{};
      std::array<T, NVARS> u_rr{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_ll[var] = u(ielem, inode - 1, var);
        u_rr[var] = u(ielem, inode, var);
      }

      auto fstar1 = NumericFlux::numerical_flux(eq, u_ll, u_rr);
      for (std::size_t var = 0; var < NVARS; ++var) {
        fstar1_L[inode][var] = fstar1[var];
        fstar1_R[inode][var] = fstar1[var];
      }
    }

    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, inode, var) =
            du(ielem, inode, var) +
            (alpha * (Basis::inv_weights[inode] *
                      (fstar1_L[inode + 1][var] - fstar1_R[inode][var])));
      }
    }
  }
};

} // namespace detail

template <class Basis, class Equations>
struct VolumeIntegralWeakForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr void operator()(std::size_t ielem, const Equations &eq,
                                   const xt::xarray<value_type> &u,
                                   xt::xarray<value_type> &du) {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel<Basis, Equations>(ielem, eq, u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations &eq,
                                         const ArrayU &u, ArrayDu &du) const {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel_kokkos<Basis, Equations>(ielem, eq, u, du);
  }
};

template <class Basis, class Equations, template <class> class NumericFlux>
struct VolumeIntegralSplitForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr void operator()(std::size_t ielem, const Equations &eq,
                                   const xt::xarray<value_type> &u,
                                   xt::xarray<value_type> &du) {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel<Basis,
                                                                   Equations>(
        ielem, eq, u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations &eq,
                                         const ArrayU &u, ArrayDu &du) const {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel_kokkos<Basis,
                                                                          Equations>(
        ielem, eq, u, du);
  }
};

struct VolumeIntegralShockCapturingBase {};

template <class Basis, class Equations, template <class> class NumericFlux,
          template <class> class FvFlux, class Indicator>
struct VolumeIntegralShockCapturingHG
    : public VolumeIntegralShockCapturingBase {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeIntegralShockCapturingHG() = delete;

  VolumeIntegralShockCapturingHG(value_type alpha_max_, value_type alpha_min_,
                                 bool alpha_smooth_,
                                 std::size_t total_elements_)
      : indicator(alpha_max_, alpha_min_, alpha_smooth_) {
    value_type eps_val = std::numeric_limits<value_type>::epsilon();
    atol = std::max(100.0 * eps_val, std::pow(eps_val, 0.75));
    alpha_view =
        Kokkos::View<value_type *, Device>("alpha_view", total_elements_);
  }

  inline void calc_alpha(std::size_t nelem, const xt::xarray<value_type> &u) {
    alpha = indicator(nelem, u);
    sync_alpha_view();
  }

  inline void sync_alpha_view() {
    auto host_alpha = Kokkos::create_mirror_view(alpha_view);
    for (std::size_t i = 0; i < alpha.size(); ++i) {
      host_alpha(i) = alpha[i];
    }
    Kokkos::deep_copy(alpha_view, host_alpha);
  }

  inline constexpr void operator()(std::size_t ielem, const Equations &eq,
                                   const xt::xarray<value_type> &u,
                                   xt::xarray<value_type> &du) {
    value_type alpha_ielem = alpha[ielem];
    bool dg_only = std::abs(alpha_ielem - 0.0) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<Basis,
                                                                     Equations>(
          ielem, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<Basis,
                                                                     Equations>(
          ielem, eq, u, du, (1.0 - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>, NDIMS>::
          template fv_kernel<Basis, Equations>(ielem, eq, u, du, alpha_ielem);
    }
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations &eq,
                                         const ArrayU &u, ArrayDu &du) const {
    value_type alpha_ielem = alpha_view(ielem);
    bool dg_only = std::abs(alpha_ielem - 0.0) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel_kokkos<Basis,
                                                                            Equations>(
          ielem, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel_kokkos<Basis,
                                                                            Equations>(
          ielem, eq, u, du, (1.0 - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>, NDIMS>::
          template fv_kernel_kokkos<Basis, Equations>(ielem, eq, u, du,
                                                      alpha_ielem);
    }
  }

private:
  Indicator indicator;
  std::vector<value_type> alpha;
  Kokkos::View<value_type *, Device> alpha_view;
  value_type atol;
};
} // namespace DGSEM
