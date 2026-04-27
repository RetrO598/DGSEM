#pragma once

#include "Kokkos_Core_fwd.hpp"
#include "base/mapping.hpp"
#include "containers/data_container.hpp"
#include "fwd/Kokkos_Fwd_CUDA.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <boundary_condition/boundary_condition.hpp>
#include <cmath>
#include <concepts>
#include <containers/containers.hpp>
#include <cstddef>
#include <cuda_runtime.h>
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
      : eq(eq_), mesh(mesh_), element(element_), boundary_set(boundary_set_),
        n_dofs(std::pow(Basis::NNodes, NDIMS)) {
    n_cells = mesh.get_num_cells();
  }

  StructuredSolver(const Equations& eq_, const Mesh& mesh_,
                   ElementCache& element_, const BoundarySetType& boundary_set_,
                   const std::array<bool, NDIMS>& periodic)
      : eq(eq_), mesh(mesh_), boundary_set(boundary_set_),
        n_dofs(std::pow(Basis::NNodes, NDIMS)) {
    n_cells = mesh.get_num_cells();
    StructuredElementInitializer<
        value_type, Basis, LinearMapping<std::array<value_type, NDIMS>>, NDIMS>
        initializer{LinearMapping<std::array<value_type, NDIMS>>(
                        mesh.get_domain_left(), mesh.get_domain_right()),
                    periodic};
    initializer.init_elements(n_cells, element_);
    element = element_;
  }

  template <class Derived>
  void initialize(const AbstractInitial<Derived, Equations>& initial_condition,
                  solution& sol);

  void calc_volume_integral(solution& sol, Kokkos::Cuda& stream);

  void calc_volume_integral(solution& sol, Kokkos::Cuda& stream)
    requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>;

  void calc_interface_flux(solution& sol);

  void calc_surface_integral(solution& sol);

  void calc_gradient_volume_integral(solution& sol, Kokkos::Cuda& stream)
    requires ParabolicEquations<Equations>;

  void transform_gradient_to_physical(solution& sol, Kokkos::Cuda& stream)
    requires ParabolicEquations<Equations>;

  void calc_gradient_interface_flux(solution& sol)
    requires ParabolicEquations<Equations>;

  void calc_gradient_surface_integral(solution& sol)
    requires ParabolicEquations<Equations>;

  void apply_gradient_jacobian(solution& sol)
    requires ParabolicEquations<Equations>;

  void calc_gradients(solution& sol)
    requires ParabolicEquations<Equations>;

  void calc_viscous_flux(solution& sol)
    requires ParabolicEquations<Equations>;

  void calc_viscous_volume_integral(solution& sol, Kokkos::Cuda& stream)
    requires ParabolicEquations<Equations>;

  void calc_viscous_interface_flux(solution& sol)
    requires ParabolicEquations<Equations>;

  void calc_viscous_surface_integral(solution& sol)
    requires ParabolicEquations<Equations>;

  // void apply_viscous_jacobian(solution& sol)
  //   requires ParabolicEquations<Equations>;

  // void accumulate_viscous_rhs(solution& sol)
  //   requires ParabolicEquations<Equations>;

  void calc_viscous_rhs(solution& sol)
    requires ParabolicEquations<Equations>;

  void apply_boundary_condition(solution& sol, value_type time);

  void apply_jacobian(solution& sol);

  void calc_rhs(solution& sol, value_type time);

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
  value_type alpha_max = static_cast<value_type>(0.5);
  value_type alpha_min = static_cast<value_type>(0.001);
  bool alpha_smooth = false;
};

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
template <class Derived>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::
    initialize(const AbstractInitial<Derived, Equations>& initial_condition,
               solution& sol) {
  InitialFunctor<value_type, Derived, NDIMS>::apply(
      sol.u_device, element.node_coordinates_device,
      static_cast<const Derived&>(initial_condition), n_cells, n_dofs, NVARS);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_volume_integral(solution& sol,
                                                             Kokkos::Cuda&
                                                                 stream) {
  if constexpr (NDIMS >= 2) {
    VolumeFlux volume_integral(element.contravariant_vectors_device);
    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells, stream);
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
                      BoundarySetType>::calc_volume_integral(solution& sol,
                                                             Kokkos::Cuda&
                                                                 stream)
  requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>
{
  if constexpr (NDIMS >= 2) {
    VolumeFlux volume_integral(alpha_max, alpha_min, alpha_smooth, n_cells,
                               element.contravariant_vectors_device,
                               element.subcell_normals.device_data());

    volume_integral.calc_alpha(n_cells, sol.u_device);
    VolumeIntegralFunctor<value_type, Equations, Basis, VolumeFlux,
                          NDIMS>::apply(sol.u_device, sol.du_device, eq,
                                        volume_integral, n_cells, stream);
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
void StructuredSolver<
    Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
    BoundarySetType>::calc_gradient_volume_integral(solution& sol,
                                                    Kokkos::Cuda& stream)
  requires ParabolicEquations<Equations>
{
  // Kokkos::deep_copy(sol.gradient_reference_device, value_type{0.0});
  GradientVolumeIntegralFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.u_device, sol.gradient_device, eq, n_cells, stream);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<
    Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
    BoundarySetType>::transform_gradient_to_physical(solution& sol,
                                                     Kokkos::Cuda& stream)
  requires ParabolicEquations<Equations>
{
  GradientPhysicalTransformFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.gradient_device, element.contravariant_vectors_device, n_cells,
      stream);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_gradient_interface_flux(solution&
                                                                         sol)
  requires ParabolicEquations<Equations>
{
  Kokkos::deep_copy(sol.gradient_surface_flux_device, value_type{0.0});
  GradientInterfaceFluxFunctor<value_type, Basis, Equations, NDIMS>::apply(
      element.left_neighbors_device, eq, sol.u_device,
      sol.gradient_surface_flux_device, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_gradient_surface_integral(solution&
                                                                           sol)
  requires ParabolicEquations<Equations>
{
  GradientSurfaceIntegralFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.gradient_device, sol.gradient_surface_flux_device,
      element.contravariant_vectors_device, element.inverse_jacobian_device,
      n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_gradient_jacobian(solution& sol)
  requires ParabolicEquations<Equations>
{
  GradientJacobianProjFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.gradient_device, element.inverse_jacobian_device, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_gradients(solution& sol)
  requires ParabolicEquations<Equations>
{
  Kokkos::deep_copy(sol.gradient_device, value_type{0.0});
  calc_gradient_volume_integral(sol);
  transform_gradient_to_physical(sol);
  calc_gradient_interface_flux(sol);
  calc_gradient_surface_integral(sol);
  apply_gradient_jacobian(sol);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_viscous_flux(solution& sol)
  requires ParabolicEquations<Equations>
{
  ViscousFluxFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.u_device, sol.gradient_device, sol.viscous_flux_device, eq, n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<
    Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
    BoundarySetType>::calc_viscous_volume_integral(solution& sol,
                                                   Kokkos::Cuda& stream)
  requires ParabolicEquations<Equations>
{
  ViscousVolumeIntegralFunctor<value_type, Basis, Equations, NDIMS>::apply(
      sol.du_device, sol.viscous_flux_device,
      element.contravariant_vectors_device, n_cells, stream);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_viscous_interface_flux(solution&
                                                                        sol)
  requires ParabolicEquations<Equations>
{
  Kokkos::deep_copy(sol.viscous_surface_flux_value_device, value_type{0.0});
  ViscousInterfaceFluxFunctor<value_type, Basis, Equations, NDIMS>::apply(
      element.left_neighbors_device, sol.viscous_flux_device,
      sol.viscous_surface_flux_value_device,
      element.contravariant_vectors_device, element.inverse_jacobian_device,
      n_cells);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_viscous_surface_integral(solution&
                                                                          sol)
  requires ParabolicEquations<Equations>
{
  SurfaceIntegralFunctor<Basis, Equations, ElementCache, NDIMS>::apply(
      sol.du_device, sol.viscous_surface_flux_value_device, eq, n_cells);
}

// template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
//           class Mesh, class BoundarySetType>
// void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
//                       BoundarySetType>::apply_viscous_jacobian(solution& sol)
//   requires ParabolicEquations<Equations>
// {
//   ParabolicJacobianProjFunctor<value_type, Basis, Equations, NDIMS>::apply(
//       sol.du_device, element.inverse_jacobian_device, n_cells);
// }

// template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
//           class Mesh, class BoundarySetType>
// void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
//                       BoundarySetType>::accumulate_viscous_rhs(solution& sol)
//   requires ParabolicEquations<Equations>
// {
//   static_assert(NDIMS == 3,
//                 "Parabolic accumulation is currently implemented for 3D
//                 only.");
//   auto du = sol.du_device;
//   auto viscous_du = sol.viscous_du_device;
//   const std::size_t ndofs = n_dofs;
//   Kokkos::parallel_for(
//       "accumulate_viscous_rhs",
//       Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
//           {0, 0, 0}, {n_cells[0], n_cells[1], n_cells[2]}),
//       KOKKOS_LAMBDA(const std::size_t ielem, const std::size_t jelem,
//                     const std::size_t kelem) {
//         for (std::size_t dof = 0; dof < ndofs; ++dof) {
//           for (std::size_t var = 0; var < NVARS; ++var) {
//             du(ielem, jelem, kelem, dof, var) +=
//                 viscous_du(ielem, jelem, kelem, dof, var);
//           }
//         }
//       });
// }

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::calc_viscous_rhs(solution& sol)
  requires ParabolicEquations<Equations>
{
  // Kokkos::deep_copy(sol.viscous_du_device, value_type{0.0});
  calc_gradients(sol);
  calc_viscous_flux(sol);
  calc_viscous_volume_integral(sol);
  calc_viscous_interface_flux(sol);
  calc_viscous_surface_integral(sol);
  // apply_viscous_jacobian(sol);
  // accumulate_viscous_rhs(sol);
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh, class BoundarySetType>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh,
                      BoundarySetType>::apply_boundary_condition(solution& sol,
                                                                 value_type
                                                                     time) {
  const auto element_data = element.device_data();
  for (std::size_t i = 0; i < NDIMS; ++i) {
    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_device, element_data, sol.surface_flux_value_device,
            2 * i, time);

    boundary_set
        .template apply<Equations, SurfaceFlux, Mesh, value_type, NDIMS>(
            mesh, eq, sol.u_device, element_data, sol.surface_flux_value_device,
            2 * i + 1, time);
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
                      BoundarySetType>::calc_rhs(solution& sol,
                                                 value_type time) {

  // create two native CUDA streams and wrap them with Kokkos::Cuda so kernels
  // scheduled with these execution spaces run on separate CUDA streams
  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  Kokkos::Cuda stream1(s1);
  Kokkos::Cuda stream2(s2);
  Kokkos::deep_copy(stream1, sol.du_device, value_type{0.0});

  calc_volume_integral(sol, stream1);

  calc_interface_flux(sol);

  apply_boundary_condition(sol, time);

  // calc_surface_integral(sol);

  // apply_jacobian(sol);

  if constexpr (ParabolicEquations<Equations>) {
    // calc_viscous_rhs(sol);
    // calc_gradients(sol);
    Kokkos::deep_copy(stream2, sol.gradient_device, value_type{0.0});
    calc_gradient_volume_integral(sol, stream2);
    transform_gradient_to_physical(sol, stream2);
    calc_gradient_interface_flux(sol);

    Kokkos::fence();
    calc_gradient_surface_integral(sol);
    apply_gradient_jacobian(sol);

    calc_viscous_flux(sol);

    Kokkos::fence();
    calc_viscous_volume_integral(sol, stream1);
    calc_viscous_interface_flux(sol);

    Kokkos::fence();
    calc_viscous_surface_integral(sol);
  }

  calc_surface_integral(sol);

  apply_jacobian(sol);

  // destroy native streams
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}

} // namespace DGSEM
