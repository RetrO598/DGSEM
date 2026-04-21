#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/utils.hpp>

namespace DGSEM {
template <class Basis, class Equations, class Analyzer, class Solution,
          class ScalarArray>
struct VolumeWeightedAnalyzerFunctor {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using DataArray = typename solution_type_traits<T, traits::NDIMS>::DataArray;

  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeWeightedAnalyzerFunctor(const Solution& sol_,
                                const ScalarArray& inverse_jacobian_)
      : u_device(sol_.u_device), inverse_jacobian(inverse_jacobian_) {}

  static void apply(Analyzer& analyzer_, const Solution& sol_,
                    const ScalarArray& inverse_jacobian_,
                    std::array<std::size_t, traits::NDIMS> n_elems_)
    requires(traits::NDIMS == 1)
  {
    VolumeWeightedAnalyzerFunctor<Basis, Equations, Analyzer, Solution,
                                  ScalarArray>
        functor(sol_, inverse_jacobian_);
    Kokkos::parallel_reduce("volume_weighted_analyzer",
                            n_elems_[0] * Basis::NNodes, functor, analyzer_);
  }

  static void apply(Analyzer& analyzer_, const Solution& sol_,
                    const ScalarArray& inverse_jacobian_,
                    std::array<std::size_t, traits::NDIMS> n_elems_)
    requires(traits::NDIMS == 2)
  {
    VolumeWeightedAnalyzerFunctor<Basis, Equations, Analyzer, Solution,
                                  ScalarArray>
        functor(sol_, inverse_jacobian_);
    Kokkos::parallel_reduce(
        "volume_weighted_analyzer",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {n_elems_[0] * Basis::NNodes, n_elems_[1] * Basis::NNodes}),
        functor, analyzer_);
  }

  static void apply(Analyzer& analyzer_, const Solution& sol_,
                    const ScalarArray& inverse_jacobian_,
                    std::array<std::size_t, traits::NDIMS> n_elems_)
    requires(traits::NDIMS == 3)
  {
    VolumeWeightedAnalyzerFunctor<Basis, Equations, Analyzer, Solution,
                                  ScalarArray>
        functor(sol_, inverse_jacobian_);
    Kokkos::parallel_reduce(
        "volume_weighted_analyzer",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0] * Basis::NNodes, n_elems_[1] * Basis::NNodes,
                        n_elems_[2] * Basis::NNodes}),
        functor, analyzer_);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         Analyzer& analyzer_) const
    requires(traits::NDIMS == 1)
  {
    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t node = idof % Basis::NNodes;

    Kokkos::Array<T, NVARS> u{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      u[var] = u_device(ielem, node, var);
    }

    const T volume_weight =
        Basis::weights(node) / Kokkos::abs(inverse_jacobian(ielem, node));
    analyzer_(u, volume_weight);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         Analyzer& analyzer_) const
    requires(traits::NDIMS == 2)
  {
    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j);

    Kokkos::Array<T, NVARS> u{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      u[var] = u_device(ielem, jelem, node, var);
    }

    const T quadrature_weight = Basis::weights(node_i) * Basis::weights(node_j);
    const T volume_weight =
        quadrature_weight /
        Kokkos::abs(inverse_jacobian(ielem, jelem, node));
    analyzer_(u, volume_weight);
  }

  KOKKOS_INLINE_FUNCTION void
  operator()(const std::size_t& idof, const std::size_t& jdof,
             const std::size_t& kdof, Analyzer& analyzer_) const
    requires(traits::NDIMS == 3)
  {
    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t kelem = kdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node_k = kdof % Basis::NNodes;
    const std::size_t node =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j, node_k);

    Kokkos::Array<T, NVARS> u{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      u[var] = u_device(ielem, jelem, kelem, node, var);
    }

    const T quadrature_weight = Basis::weights(node_i) * Basis::weights(node_j) *
                                Basis::weights(node_k);
    const T volume_weight =
        quadrature_weight /
        Kokkos::abs(inverse_jacobian(ielem, jelem, kelem, node));
    analyzer_(u, volume_weight);
  }

  KOKKOS_INLINE_FUNCTION void init(Analyzer& analyzer_) const {
    analyzer_.init();
  }

  KOKKOS_INLINE_FUNCTION void join(Analyzer& dst, const Analyzer& src) const {
    dst.join(src);
  }

  DataArray u_device;
  ScalarArray inverse_jacobian;
};
} // namespace DGSEM
