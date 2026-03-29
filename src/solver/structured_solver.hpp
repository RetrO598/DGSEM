#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <boundary_condition/boundary_condition.hpp>
#include <cmath>
#include <concepts>
#include <containers/containers.hpp>
#include <cstddef>
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
  StructuredSolver(const Equations& eq_, const Mesh& mesh_,
                   const ElementCache& element_,
                   const BoundarySetType& boundary_set_)
      : eq(eq_),
        mesh(mesh_),
        element(element_),
        boundary_set(boundary_set_),
        n_dofs(std::pow(Basis::NNodes, NDIMS)) {
    n_cells = mesh.get_num_cells();
  }

  template <class Derived>
  void initialize(const AbstractInitial<Derived, Equations>& initial_condition,
                  solution& sol);

  void calc_volume_integral(solution& sol);

  void calc_volume_integral(solution& sol)
    requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>;

  void calc_interface_flux(solution& sol);

  void calc_surface_integral(solution& sol);

  void apply_boundary_condition(solution& sol);

  void apply_jacobian(solution& sol);

  void calc_rhs(solution& sol);

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
  std::array<std::size_t, NDIMS> n_cells;

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
    initialize(const AbstractInitial<Derived, Equations>& initial_condition,
               solution& sol) {
  InitialFunctor<value_type, AbstractInitial<Derived, Equations>, NDIMS>::apply(
      sol.u_device, element.node_coordinates_device, initial_condition, n_cells,
      n_dofs, NVARS);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_volume_integral(solution& sol) {
  if constexpr (NDIMS == 2) {
    VolumeFlux volume_integral(element.contravariant_vectors_device);
    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells);
  } else {
    VolumeFlux volume_integral{};

    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_volume_integral(solution& sol)
  requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>
{
  if constexpr (NDIMS == 2) {
    VolumeFlux volume_integral(alpha_max, alpha_min, alpha_smooth, n_cells,
                               element.contravariant_vectors_device);

    volume_integral.calc_alpha(n_cells, sol.u_device);
    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells);
  } else {
    VolumeFlux volume_integral(alpha_max, alpha_min, alpha_smooth, n_cells);

    volume_integral.calc_alpha(n_cells, sol.u_device);
    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_interface_flux(solution& sol) {
  InterfaceFluxFunctor<
      value_type, Equations, Basis,
      InterfaceHelper<Basis, Equations, SurfaceFlux, ElementCache>,
      NDIMS>::apply(element.left_neighbors_device,
                    element.contravariant_vectors_device,
                    element.inverse_jacobian_device, eq, sol.u_device,
                    sol.surface_flux_value_device, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_surface_integral(solution& sol) {
  SurfaceIntegralFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_device, sol.surface_flux_value_device, eq, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_boundary_condition(solution&
                                                                     sol) {
  for (std::size_t i = 0; i < NDIMS; ++i) {
    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_device, sol.surface_flux_value_device, 2 * i, 0.0);

    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_device, sol.surface_flux_value_device, 2 * i + 1,
            0.0);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_jacobian(solution& sol) {
  JacobianProjFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_device, element.inverse_jacobian_device, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_rhs(solution& sol) {
  Kokkos::deep_copy(sol.du_device, 0.0);

  calc_volume_integral(sol);

  calc_interface_flux(sol);

  apply_boundary_condition(sol);

  calc_surface_integral(sol);

  apply_jacobian(sol);
}

}  // namespace DGSEM
