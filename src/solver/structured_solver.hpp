#pragma once
#include "boundary_condition/initial_condition_base.hpp"
#include "boundary_condition/initial_functor.hpp"
#include "space_integral/jacobian.hpp"
#include "space_integral/surface_flux.hpp"
#include "space_integral/surface_flux_functor.hpp"
#include "space_integral/surface_integral_functor.hpp"
#include "space_integral/volume_integral_functor.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
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

  // std::size_t total_elements = mesh.get_nelem();
  // std::size_t n_dofs = std::pow(Basis::NNodes, NDIMS);

  // std::array<value_type, NDIMS> coord;
  // for (std::size_t elem = 0; elem < total_elements; ++elem) {
  //   for (std::size_t inode = 0; inode < n_dofs; ++inode) {
  //     for (std::size_t idim = 0; idim < NDIMS; ++idim) {
  //       coord[idim] = element.node_coordinates(elem, inode, idim);
  //     }

  //     auto init_value = initial_condition.get_initial(coord);
  //     for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
  //       sol.u(elem, inode, ivar) = init_value[ivar];
  //     }
  //   }
  // }

  // auto init = initial_condition;
  // auto u = sol.u_kokkos;
  // auto node_coords = element.node_coordinates_kokkos;

  // Kokkos::parallel_for(
  //     "initialize_elements", total_elements, KOKKOS_LAMBDA(const int elem) {
  //       std::array<value_type, NDIMS> coord;
  //       for (std::size_t inode = 0; inode < nodes; ++inode) {
  //         for (std::size_t idim = 0; idim < NDIMS; ++idim) {
  //           coord[idim] = node_coords(elem, inode, idim);
  //         }

  //         auto init_value = init.get_initial(coord);

  //         for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
  //           u(elem, inode, ivar) = init_value[ivar];
  //         }
  //       }
  //     });

  // InitialFunctor<value_type, AbstractInitial<Derived, Equations>,
  // NDIMS>::apply(
  //     sol.u_kokkos, element.node_coordinates_kokkos, init, total_elements,
  //     nodes, NVARS);

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
  // for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
  //   volume_integral(ielem, eq, sol.u, sol.du);
  // }

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
  // volume_integral.calc_alpha(total_elements, sol.u);
  volume_integral.calc_alpha_kokkos(total_elements, sol.u_kokkos);
  // for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
  //   volume_integral(ielem, eq, sol.u, sol.du);
  // }

  VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux, NDIMS>::apply(
      sol.u_kokkos, sol.du_kokkos, eq, volume_integral, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_interface_flux(solution &sol) {
  // std::size_t total_elements = mesh.get_nelem();
  // for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
  //   InterfaceHelper<Basis, Equations, SurfaceFlux,
  //                   ElementCache>::interface_flux(element, eq, ielem, sol.u,
  //                                                 sol.surface_flux_value);
  // }

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
  // std::size_t total_elements = mesh.get_nelem();
  // for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
  //   SurfaceIntegral<Basis, Equations, ElementCache>::integral(
  //       element, ielem, sol.du, sol.surface_flux_value);
  // }

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
  // std::size_t total_elements = mesh.get_nelem();
  // for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
  //   JacobianProj<Basis, Equations, ElementCache>::apply(element, ielem,
  //   sol.du);
  // }

  JacobianProjFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_kokkos, element.inverse_jacobian_kokkos, mesh.get_num_cells());
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_rhs(solution &sol) {
  // sol.du.fill(0.0); // Clear the residual before accumulation
  Kokkos::deep_copy(sol.du_kokkos, 0.0);
  calc_volume_integral(sol);
  // sol.check_solution();
  calc_interface_flux(sol);
  // sol.check_solution();
  apply_boundary_condition(sol);
  // sol.check_solution();
  calc_surface_integral(sol);
  // sol.check_solution();
  apply_jacobian(sol);
  // sol.check_solution();
}

} // namespace DGSEM
