#pragma once
#include "boundary_condition.hpp"
#include "data_container.hpp"
#include "equations.hpp"
#include "initial_condition.hpp"
#include "jacobian.hpp"
#include "surface_flux.hpp"
#include "surface_integral.hpp"
#include "volume_flux.hpp"
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace DGSEM {

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
class StructuredSolver {
public:
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using ElementCache = StructuredElementContainer<value_type, NDIMS>;

  StructuredSolver() = delete;
  StructuredSolver(const Equations &eq_, const Mesh &mesh_,
                   const ElementCache &element_)
      : eq(eq_), mesh(mesh_), element(element_) {}

  template <class Derived>
  void initialize(const AbstractInitial<Derived, Equations> &initial_condition,
                  Solution<Mesh, Basis, Equations> &sol);

  void calc_volume_integral(Solution<Mesh, Basis, Equations> &sol);

  void calc_volume_integral(Solution<Mesh, Basis, Equations> &sol)
    requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>;

  void calc_interface_flux(Solution<Mesh, Basis, Equations> &sol);

  void calc_surface_integral(Solution<Mesh, Basis, Equations> &sol);

  void apply_boundary_condition(Solution<Mesh, Basis, Equations> &sol);

  void apply_jacobian(Solution<Mesh, Basis, Equations> &sol);

  void calc_rhs(Solution<Mesh, Basis, Equations> &sol);

  void set_indicator_parameters(value_type alpha_max_, value_type alpha_min_,
                                bool alpha_smooth_) {
    alpha_max = alpha_max_;
    alpha_min = alpha_min_;
    alpha_smooth = alpha_smooth_;
  }

private:
  Equations eq;
  Mesh mesh;
  ElementCache element;

  // for shock-capturing indicators
  value_type alpha_max = 0.5;
  value_type alpha_min = 0.001;
  bool alpha_smooth = false;
};

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
template <class Derived>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    initialize(const AbstractInitial<Derived, Equations> &initial_condition,
               Solution<Mesh, Basis, Equations> &sol) {

  std::size_t total_elements = mesh.get_nelem();
  std::size_t nodes = std::pow(Basis::NNodes, NDIMS);

  std::array<value_type, NDIMS> coord;
  for (std::size_t elem = 0; elem < total_elements; ++elem) {
    for (std::size_t inode = 0; inode < nodes; ++inode) {
      for (std::size_t idim = 0; idim < NDIMS; ++idim) {
        coord[idim] = element.node_coordinates(elem, inode, idim);
      }

      auto init_value = initial_condition.get_initial(coord);
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        sol.u(elem, inode, ivar) = init_value[ivar];
      }
    }
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    calc_volume_integral(Solution<Mesh, Basis, Equations> &sol) {
  std::size_t total_elements = mesh.get_nelem();
  VolumeFlux volume_integral{};
  for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
    volume_integral(ielem, eq, sol.u, sol.du);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    calc_volume_integral(Solution<Mesh, Basis, Equations> &sol)
  requires std::derived_from<VolumeFlux, VolumeIntegralShockCapturingBase>
{
  std::size_t total_elements = mesh.get_nelem();
  VolumeFlux volume_integral(alpha_max, alpha_min, alpha_smooth,
                             total_elements);
  volume_integral.calc_alpha(total_elements, sol.u);
  for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
    volume_integral(ielem, eq, sol.u, sol.du);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    calc_interface_flux(Solution<Mesh, Basis, Equations> &sol) {
  std::size_t total_elements = mesh.get_nelem();
  for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
    InterfaceHelper<Basis, Equations, SurfaceFlux,
                    ElementCache>::interface_flux(element, eq, ielem, sol.u,
                                                  sol.surface_flux_value);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    calc_surface_integral(Solution<Mesh, Basis, Equations> &sol) {
  std::size_t total_elements = mesh.get_nelem();
  for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
    SurfaceIntegral<Basis, Equations, ElementCache>::integral(
        element, ielem, sol.du, sol.surface_flux_value);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    apply_boundary_condition(Solution<Mesh, Basis, Equations> &sol) {
  for (std::size_t i = 0; i < NDIMS; ++i) {
    auto bc1 = mesh.get_boundary(i);
    auto bc2 = mesh.get_boundary(i + NDIMS);

    BoundaryDispatcher<Basis, Equations, NDIMS>::apply(
        bc1, mesh.get_num_cells(), i, sol.u, sol.surface_flux_value);

    BoundaryDispatcher<Basis, Equations, NDIMS>::apply(
        bc2, mesh.get_num_cells(), i + NDIMS, sol.u, sol.surface_flux_value);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux, Mesh>::
    apply_jacobian(Solution<Mesh, Basis, Equations> &sol) {
  std::size_t total_elements = mesh.get_nelem();
  for (std::size_t ielem = 0; ielem < total_elements; ++ielem) {
    JacobianProj<Basis, Equations, ElementCache>::apply(element, ielem, sol.du);
  }
}

template <class Equations, class Basis, class VolumeFlux, class SurfaceFlux,
          class Mesh>
void StructuredSolver<Equations, Basis, VolumeFlux, SurfaceFlux,
                      Mesh>::calc_rhs(Solution<Mesh, Basis, Equations> &sol) {
  sol.du.fill(0.0); // Clear the residual before accumulation
  calc_volume_integral(sol);
  calc_interface_flux(sol);
  calc_surface_integral(sol);
  apply_boundary_condition(sol);
  apply_jacobian(sol);
}

} // namespace DGSEM
