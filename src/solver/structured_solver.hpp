#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <boundary_condition/boundary_condition.hpp>
#include <cmath>
#include <concepts>
#include <containers/containers.hpp>
#include <cstddef>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <equations/equations.hpp>
#include <space_integral/space_integral.hpp>

namespace DGSEM {

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
class StructuredSolver {
public:
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using EquationType = Equations;

  using ElementCache = StructuredElementContainer<value_type, NDIMS>;

  using solution = Solution<Mesh, Basis, Equations>;

  StructuredSolver() = delete;
  StructuredSolver(const Equations &eq_, const Mesh &mesh_,
                   const ElementCache &element_,
                   const BoundarySetType &boundary_set_)
      : eq(eq_), mesh(mesh_), element(element_), boundary_set(boundary_set_),
        n_dofs(std::pow(Basis::NNodes, NDIMS)) {}

  template <class Derived>
  void initialize(const AbstractInitial<Derived, Equations> &initial_condition,
                  solution &sol);

  void calc_volume_integral(solution &sol);

  void calc_volume_integral(solution &sol)
    requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>;

  void calc_interface_flux(solution &sol);

  void calc_surface_integral(solution &sol);

  void apply_boundary_condition(solution &sol);

  void apply_jacobian(solution &sol);

  void calc_rhs(solution &sol);

  void set_indicator_parameters(value_type alpha_max_, value_type alpha_min_,
                                bool alpha_smooth_) {
    alpha_max = alpha_max_;
    alpha_min = alpha_min_;
    alpha_smooth = alpha_smooth_;
  }

  std::size_t get_ndofs() const { return n_dofs; }

private:
  BoundarySetType boundary_set;
  Equations eq;
  Mesh mesh;
  ElementCache element;
  std::size_t n_dofs;

  // for shock-capturing indicators
  value_type alpha_max = 0.5;
  value_type alpha_min = 0.001;
  bool alpha_smooth = false;
};

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
template <class Derived>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::
    initialize(const AbstractInitial<Derived, Equations> &initial_condition,
               solution &sol) {

  InitialFunctor<value_type, AbstractInitial<Derived, Equations>, NDIMS>::apply(
      sol.u_kokkos, element.node_coordinates_kokkos, initial_condition,
      mesh.get_num_cells(), n_dofs, NVARS);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_volume_integral(solution &sol) {
  std::size_t total_elements = mesh.get_nelem();
  VolumeFlux volume_integral{};

  VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux, NDIMS>::apply(
      sol.u_kokkos, sol.du_kokkos, eq, volume_integral, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_volume_integral(solution &sol)
  requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>
{
  std::size_t total_elements = mesh.get_nelem();
  VolumeFlux volume_integral(alpha_max, alpha_min, alpha_smooth,
                             total_elements);

  volume_integral.calc_alpha_kokkos(total_elements, sol.u_kokkos);
  VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux, NDIMS>::apply(
      sol.u_kokkos, sol.du_kokkos, eq, volume_integral, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_interface_flux(solution &sol) {

  InterfaceFluxFunctor<
      value_type, Equations, Basis,
      InterfaceHelper<Basis, Equations, SurfaceFlux, ElementCache>,
      NDIMS>::apply(element.left_neighbors_kokkos, eq, sol.u_kokkos,
                    sol.surface_flux_value_kokkos, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_surface_integral(solution &sol) {

  SurfaceIntegralFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_kokkos, sol.surface_flux_value_kokkos, eq, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_boundary_condition(solution
                                                                     &sol) {

  for (std::size_t i = 0; i < NDIMS; ++i) {
    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_kokkos, sol.surface_flux_value_kokkos, 2 * i, 0.0);

    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_kokkos, sol.surface_flux_value_kokkos, 2 * i + 1,
            0.0);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_jacobian(solution &sol) {

  JacobianProjFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_kokkos, element.inverse_jacobian_kokkos, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_rhs(solution &sol) {

  Kokkos::deep_copy(sol.du_kokkos, 0.0);
  calc_volume_integral(sol);

  calc_interface_flux(sol);

  apply_boundary_condition(sol);

  calc_surface_integral(sol);

  apply_jacobian(sol);
}

} // namespace DGSEM
