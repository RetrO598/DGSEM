#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/utils.hpp>
namespace DGSEM {
template <class Basis, class Equations, class Analyzer, class Solution>
struct AnalyzerFunctor {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using DataArray = typename solution_type_traits<T, traits::NDIMS>::DataArray;

  constexpr static std::size_t NVARS = traits::NVARS;

  AnalyzerFunctor(const Solution& sol_) : u_device(sol_.u_device) {}

  static void apply(Analyzer& analyzer_, const Solution& sol_,
                    std::array<std::size_t, traits::NDIMS> n_elems_)
    requires(traits::NDIMS == 1)
  {
    AnalyzerFunctor<Basis, Equations, Analyzer, Solution> functor(sol_);
    Kokkos::parallel_reduce("analyzer", n_elems_[0] * Basis::NNodes, functor,
                            analyzer_);
  }

  static void apply(Analyzer& analyzer_, const Solution& sol_,
                    std::array<std::size_t, traits::NDIMS> n_elems_)
    requires(traits::NDIMS == 2)
  {

    AnalyzerFunctor<Basis, Equations, Analyzer, Solution> functor(sol_);
    Kokkos::parallel_reduce(
        "analyzer",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {n_elems_[0] * Basis::NNodes, n_elems_[1] * Basis::NNodes}),
        functor, analyzer_);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         Analyzer& analyzer_) const
    requires(traits::NDIMS == 1)
  {
    std::size_t ielem = idof / Basis::NNodes;
    std::size_t node = idof % Basis::NNodes;

    Kokkos::Array<T, NVARS> u{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      u[var] = u_device(ielem, node, var);
    }
    analyzer_(u);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         Analyzer& analyzer_) const
    requires(traits::NDIMS == 2)
  {
    std::size_t ielem = idof / Basis::NNodes;
    std::size_t jelem = jdof / Basis::NNodes;
    std::size_t node_i = idof % Basis::NNodes;
    std::size_t node_j = jdof % Basis::NNodes;

    Kokkos::Array<T, NVARS> u{};
    for (std::size_t var = 0; var < NVARS; ++var) {
      std::size_t node = DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j);
      u[var] = u_device(ielem, jelem, node, var);
    }
    analyzer_(u);
  }

  KOKKOS_INLINE_FUNCTION void init(Analyzer& analyzer_) const {
    analyzer_.init();
  }

  KOKKOS_INLINE_FUNCTION void join(Analyzer& dst, const Analyzer& src) const {
    dst.join(src);
  }

  DataArray u_device;
};
} // namespace DGSEM