#pragma once

#include <array>
#include <base/mapping.hpp>
#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <mesh/mesh.hpp>
#include <solver/structured_solver.hpp>
#include <time_integrator/SSPRK3.hpp>

namespace DGSEM {

namespace detail {
template <class Equations>
using equation_value_t =
    typename equations::EquationTraits<Equations>::value_type;

template <class Equations>
inline constexpr std::size_t equation_ndims_v =
    equations::EquationTraits<Equations>::NDIMS;

template <class Equations>
using equation_domain_t =
    std::array<equation_value_t<Equations>, equation_ndims_v<Equations>>;

template <class Equations>
using equation_cells_t = std::array<std::size_t, equation_ndims_v<Equations>>;

template <class Equations>
using equation_periodic_t = std::array<bool, equation_ndims_v<Equations>>;

template <class T, std::size_t NDIMS>
struct DefaultStructuredMapping {
  using type = LinearMapping<std::array<T, NDIMS>>;

  static type make(const std::array<T, NDIMS>& domain_left,
                   const std::array<T, NDIMS>& domain_right) {
    return type(domain_left, domain_right);
  }
};

template <class T>
struct DefaultStructuredMapping<T, 1> {
  using type = LinearMapping<T>;

  static type make(const std::array<T, 1>& domain_left,
                   const std::array<T, 1>& domain_right) {
    return type(domain_left[0], domain_right[0]);
  }
};

template <std::size_t NDIMS>
constexpr std::array<bool, NDIMS> non_periodic_boundaries() {
  std::array<bool, NDIMS> periodic{};
  return periodic;
}

template <class Equations>
constexpr equation_periodic_t<Equations> default_periodic_boundaries() {
  return non_periodic_boundaries<equation_ndims_v<Equations>>();
}
} // namespace detail

template <class Equations>
using StructuredDomain = detail::equation_domain_t<Equations>;

template <class Equations>
using StructuredCellCounts = detail::equation_cells_t<Equations>;

template <class Equations>
using StructuredPeriodicFlags = detail::equation_periodic_t<Equations>;

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class BoundarySetType,
          class Mapping = typename detail::DefaultStructuredMapping<
              detail::equation_value_t<Equations>,
              detail::equation_ndims_v<Equations>>::type>
class StructuredProblem {
public:
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;

  using Mesh = StructuredMesh<value_type, NDIMS>;
  using ElementCache = StructuredElementContainer<value_type, NDIMS>;
  using Domain = StructuredDomain<Equations>;
  using CellCounts = StructuredCellCounts<Equations>;
  using PeriodicFlags = StructuredPeriodicFlags<Equations>;
  using Solver = StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux,
                                  Mesh, BoundarySetType>;
  using SolutionType = Solution<Mesh, Basis, Equations>;
  using TimeIntegrator = SSPRK3<value_type, Solver, Mesh, SolutionType>;

  StructuredProblem(const Equations& equation, const Domain& domain_left,
                    const Domain& domain_right, const CellCounts& n_cells,
                    const BoundarySetType& boundary_set,
                    const PeriodicFlags& periodic, const Mapping& mapping)
      : mesh_(domain_left, domain_right, n_cells),
        elements_(make_elements(n_cells, periodic, mapping)),
        solver_(equation, mesh_, elements_, boundary_set), solution_(mesh_) {}

  template <class InitialCondition>
  void initialize(const InitialCondition& initial_condition) {
    solver_.initialize(initial_condition, solution_);
  }

  TimeIntegrator make_ssprk3() { return TimeIntegrator(solution_, mesh_); }

  TimeIntegrator make_ssprk3(value_type final_time) {
    return TimeIntegrator(solution_, mesh_, final_time);
  }

  Mesh& mesh() { return mesh_; }
  const Mesh& mesh() const { return mesh_; }

  ElementCache& elements() { return elements_; }
  const ElementCache& elements() const { return elements_; }

  Solver& solver() { return solver_; }
  const Solver& solver() const { return solver_; }

  SolutionType& solution() { return solution_; }
  const SolutionType& solution() const { return solution_; }

private:
  static ElementCache make_elements(const CellCounts& n_cells,
                                    const PeriodicFlags& periodic,
                                    const Mapping& mapping) {
    ElementCache elements;
    StructuredElementInitializer<value_type, Basis, Mapping, NDIMS> initializer{
        mapping, periodic};
    initializer.init_elements(n_cells, elements);
    return elements;
  }

  Mesh mesh_;
  ElementCache elements_;
  Solver solver_;
  SolutionType solution_;
};

template <class Basis, class VolumeFlux, class SurfaceFlux, class Equations,
          class BoundarySetType>
auto make_structured_problem(
    const Equations& equation, const StructuredDomain<Equations>& domain_left,
    const StructuredDomain<Equations>& domain_right,
    const StructuredCellCounts<Equations>& n_cells,
    const BoundarySetType& boundary_set,
    const StructuredPeriodicFlags<Equations>& periodic =
        detail::default_periodic_boundaries<Equations>()) {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr std::size_t NDIMS = traits::NDIMS;
  using Mapping =
      typename detail::DefaultStructuredMapping<value_type, NDIMS>::type;
  return StructuredProblem<Equations, Basis, VolumeFlux, SurfaceFlux,
                           BoundarySetType, Mapping>(
      equation, domain_left, domain_right, n_cells, boundary_set, periodic,
      detail::DefaultStructuredMapping<value_type, NDIMS>::make(domain_left,
                                                                domain_right));
}

template <class Basis, class VolumeFlux, class SurfaceFlux, class Equations,
          class BoundarySetType, class Mapping>
auto make_structured_problem(const Equations& equation,
                             const StructuredDomain<Equations>& domain_left,
                             const StructuredDomain<Equations>& domain_right,
                             const StructuredCellCounts<Equations>& n_cells,
                             const BoundarySetType& boundary_set,
                             const StructuredPeriodicFlags<Equations>& periodic,
                             const Mapping& mapping) {
  return StructuredProblem<Equations, Basis, VolumeFlux, SurfaceFlux,
                           BoundarySetType, Mapping>(
      equation, domain_left, domain_right, n_cells, boundary_set, periodic,
      mapping);
}

template <class Basis, class VolumeFlux, class SurfaceFlux, class Equations,
          class BoundarySetType>
auto make_structured_problem(
    const Equations& equation,
    const detail::equation_value_t<Equations>& domain_left,
    const detail::equation_value_t<Equations>& domain_right,
    const StructuredCellCounts<Equations>& n_cells,
    const BoundarySetType& boundary_set,
    const StructuredPeriodicFlags<Equations>& periodic =
        detail::default_periodic_boundaries<Equations>())
  requires(detail::equation_ndims_v<Equations> == 1)
{
  return make_structured_problem<Basis, VolumeFlux, SurfaceFlux>(
      equation, detail::equation_domain_t<Equations>{domain_left},
      detail::equation_domain_t<Equations>{domain_right}, n_cells, boundary_set,
      periodic);
}

} // namespace DGSEM
