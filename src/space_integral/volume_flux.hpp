#pragma once

#include <algorithm>
#include <array>
#include <base/base.hpp>
#include <cmath>
#include <cstddef>
#include <equations/equations.hpp>

namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct VolumeIntegralWeak;

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void weak_form_kernel(std::size_t ielem,
                                                      const BasisData& basis,
                                                      const Equations& eq,
                                                      const ArrayU& u,
                                                      ArrayDu& du) {
    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      std::array<T, NVARS> flux;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }
      flux = eq.flux(u_node, 0);

      for (std::size_t iinode = 0; iinode < BasisData::NNodes; ++iinode) {
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, iinode, var) =
              du(ielem, iinode, var) + basis.derivative_dhat(iinode, inode) * flux[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct VolumeIntegralSplit;

template <class T, std::size_t NVARS, class NumericFlux>
struct VolumeIntegralSplit<T, NVARS, NumericFlux, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void split_form_kernel(std::size_t ielem,
                                                       const BasisData& basis,
                                                       const Equations& eq,
                                                       const ArrayU& u,
                                                       ArrayDu& du) {
    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < BasisData::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) + basis.derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) + basis.derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }

  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void split_form_kernel(std::size_t ielem,
                                                       const BasisData& basis,
                                                       const Equations& eq,
                                                       const ArrayU& u,
                                                       ArrayDu& du, T alpha) {
    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < BasisData::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              alpha * basis.derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              alpha * basis.derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct FinitVolumeIntegral;

template <class T, std::size_t NVARS, class NumericFlux>
struct FinitVolumeIntegral<T, NVARS, NumericFlux, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void fv_kernel(std::size_t ielem,
                                               const BasisData& basis,
                                               const Equations& eq,
                                               const ArrayU& u, ArrayDu& du,
                                               T alpha) {
    std::array<std::array<T, NVARS>, BasisData::NNodes + 1> fstar1_L{};
    std::array<std::array<T, NVARS>, BasisData::NNodes + 1> fstar1_R{};
    for (std::size_t inode = 1; inode < BasisData::NNodes; ++inode) {
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

    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, inode, var) =
            du(ielem, inode, var) +
            (alpha * (basis.inv_weights[inode] *
                      (fstar1_L[inode + 1][var] - fstar1_R[inode][var])));
      }
    }
  }
};

}  // namespace detail

template <class Basis, class Equations>
struct VolumeIntegralWeakForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeIntegralWeakForm() : basis_data(Basis::device_data()) {}

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::template
        weak_form_kernel<BasisData, Equations>(ielem, basis_data, eq, u, du);
  }

  BasisData basis_data;
};

template <class Basis, class Equations, template <class> class NumericFlux>
struct VolumeIntegralSplitForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeIntegralSplitForm() : basis_data(Basis::device_data()) {}

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel<BasisData,
                                                                   Equations>(
        ielem, basis_data, eq, u, du);
  }

  BasisData basis_data;
};

struct VolumeIntegralShockCapturingBase {};

template <class Basis, class Equations, template <class> class NumericFlux,
          template <class> class FvFlux, class Indicator>
struct VolumeIntegralShockCapturingHG
    : public VolumeIntegralShockCapturingBase {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using AlphaView = solution_type_traits<value_type, NDIMS>::ElementArray;
  using DataArray = solution_type_traits<value_type, NDIMS>::DataArray;

  VolumeIntegralShockCapturingHG() = delete;

  VolumeIntegralShockCapturingHG(value_type alpha_max_, value_type alpha_min_,
                                 bool alpha_smooth_,
                                 const std::array<std::size_t, NDIMS>& n_cells_)
      : indicator(alpha_max_, alpha_min_, alpha_smooth_, Basis::device_data()),
        basis_data(Basis::device_data()) {
    value_type eps_val = std::numeric_limits<value_type>::epsilon();
    atol = std::max(100.0 * eps_val, std::pow(eps_val, 0.75));

    if constexpr (AlphaView::rank == 1) {
      alpha_view = AlphaView("alpha_view", n_cells_[0]);
    } else if constexpr (AlphaView::rank == 2) {
      alpha_view = AlphaView("alpha_view", n_cells_[1], n_cells_[0]);
    } else if constexpr (AlphaView::rank == 3) {
      alpha_view =
          AlphaView("alpha_view", n_cells_[2], n_cells_[1], n_cells_[0]);
    }
  }

  inline void calc_alpha(const std::array<std::size_t, NDIMS>& n_cells,
                         DataArray u) {
    indicator(n_cells, u, alpha_view);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const {
    value_type alpha_ielem = alpha_view(ielem);
    bool dg_only = std::abs(alpha_ielem - 0.0) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, basis_data, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, basis_data, eq, u, du, (1.0 - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>, NDIMS>::
          template fv_kernel<BasisData, Equations>(ielem, basis_data, eq, u, du,
                                                   alpha_ielem);
    }
  }

 private:
  Indicator indicator;
  AlphaView alpha_view;
  BasisData basis_data;
  value_type atol;
};
}  // namespace DGSEM
