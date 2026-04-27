#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>
#include <space_integral/parabolic_concepts.hpp>

namespace DGSEM {

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct ViscousFluxFunctor;

template <class T, class Basis, class Equations>
struct ViscousFluxFunctor<T, Basis, Equations, 3> {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;

  ViscousFluxFunctor(DataArray u_, VectorFieldArray gradient_,
                     VectorFieldArray viscous_flux_, const Equations& eq_)
      : u(u_), gradient(gradient_), viscous_flux(viscous_flux_), eq(eq_) {}

  static void apply(DataArray u_, VectorFieldArray gradient_,
                    VectorFieldArray viscous_flux_, const Equations& eq_,
                    const std::array<std::size_t, 3>& n_elems_) {
    ViscousFluxFunctor functor(u_, gradient_, viscous_flux_, eq_);
    Kokkos::parallel_for(
        "viscous_flux",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0], n_elems_[1], n_elems_[2]}),
        functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const {
    constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes * Basis::NNodes;
    for (std::size_t dof = 0; dof < ndofs; ++dof) {
      std::array<T, traits::NVARS> state{};
      for (std::size_t var = 0; var < traits::NVARS; ++var) {
        state[var] = u(ielem, jelem, kelem, dof, var);
      }
      const auto q = eq.gradient_variables(state);

      std::array<std::array<T, Equations::NGRAD_VARS>, 3> grad_q{};
      for (std::size_t dim = 0; dim < 3; ++dim) {
        for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
          grad_q[dim][var] = gradient(ielem, jelem, kelem, dof, var, dim);
        }
      }

      for (std::size_t dim = 0; dim < 3; ++dim) {
        const auto flux = eq.viscous_flux(q, grad_q, dim);
        for (std::size_t var = 0; var < traits::NVARS; ++var) {
          viscous_flux(ielem, jelem, kelem, dof, var, dim) = flux[var];
        }
      }
    }
  }

  DataArray u;
  VectorFieldArray gradient;
  VectorFieldArray viscous_flux;
  Equations eq;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct ViscousVolumeIntegralFunctor;

template <class T, class Basis, class Equations>
struct ViscousVolumeIntegralFunctor<T, Basis, Equations, 3> {
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using MetricArray = typename jacobian_type_traits<T, 3>::JacobianMatrix;
  using BasisData = typename Basis::DeviceData;

  ViscousVolumeIntegralFunctor(DataArray du_, VectorFieldArray viscous_flux_,
                               MetricArray contravariant_vectors_)
      : du(du_), viscous_flux(viscous_flux_),
        contravariant_vectors(contravariant_vectors_),
        basis(Basis::device_data()) {}

  static void apply(DataArray du_, VectorFieldArray viscous_flux_,
                    MetricArray contravariant_vectors_,
                    const std::array<std::size_t, 3>& n_elems_) {
    ViscousVolumeIntegralFunctor functor(du_, viscous_flux_,
                                         contravariant_vectors_);
    Kokkos::parallel_for(
        "viscous_volume_integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0], n_elems_[1], n_elems_[2]}),
        functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const {
    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
        for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
          const std::size_t dof =
              DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, knode);

          const T ja11 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 0);
          const T ja12 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 1);
          const T ja13 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 2);
          const T ja21 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 0);
          const T ja22 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 1);
          const T ja23 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 2);
          const T ja31 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 0);
          const T ja32 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 1);
          const T ja33 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 2);

          std::array<T, Equations::NVARS> flux_xi{};
          std::array<T, Equations::NVARS> flux_eta{};
          std::array<T, Equations::NVARS> flux_zeta{};
          for (std::size_t var = 0; var < Equations::NVARS; ++var) {
            const T f1 = viscous_flux(ielem, jelem, kelem, dof, var, 0);
            const T f2 = viscous_flux(ielem, jelem, kelem, dof, var, 1);
            const T f3 = viscous_flux(ielem, jelem, kelem, dof, var, 2);
            flux_xi[var] = ja11 * f1 + ja12 * f2 + ja13 * f3;
            flux_eta[var] = ja21 * f1 + ja22 * f2 + ja23 * f3;
            flux_zeta[var] = ja31 * f1 + ja32 * f2 + ja33 * f3;
          }

          for (std::size_t ii = 0; ii < Basis::NNodes; ++ii) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(ii, jnode, knode);
            for (std::size_t var = 0; var < Equations::NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(ii, inode) * flux_xi[var];
            }
          }

          for (std::size_t jj = 0; jj < Basis::NNodes; ++jj) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jj, knode);
            for (std::size_t var = 0; var < Equations::NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(jj, jnode) * flux_eta[var];
            }
          }

          for (std::size_t kk = 0; kk < Basis::NNodes; ++kk) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, kk);
            for (std::size_t var = 0; var < Equations::NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(kk, knode) * flux_zeta[var];
            }
          }
        }
      }
    }
  }

  DataArray du;
  VectorFieldArray viscous_flux;
  MetricArray contravariant_vectors;
  BasisData basis;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct ViscousInterfaceFluxFunctor;

template <class T, class Basis, class Equations>
struct ViscousInterfaceFluxFunctor<T, Basis, Equations, 3> {
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using IndexArray = typename index_type_traits<3>::IndexArray;
  using MetricArray = typename jacobian_type_traits<T, 3>::JacobianMatrix;
  using ScalarArray = typename scalar_node_type_traits<T, 3>::ScalarArray;

  ViscousInterfaceFluxFunctor(IndexArray neighbors_,
                              VectorFieldArray viscous_flux_,
                              DataArray surface_flux_,
                              MetricArray contravariant_vectors_,
                              ScalarArray inverse_jacobian_)
      : neighbors(neighbors_), viscous_flux(viscous_flux_),
        surface_flux(surface_flux_),
        contravariant_vectors(contravariant_vectors_),
        inverse_jacobian(inverse_jacobian_) {}

  static void apply(IndexArray neighbors_, VectorFieldArray viscous_flux_,
                    DataArray surface_flux_, MetricArray contravariant_vectors_,
                    ScalarArray inverse_jacobian_,
                    const std::array<std::size_t, 3>& n_elems_) {
    ViscousInterfaceFluxFunctor functor(neighbors_, viscous_flux_,
                                        surface_flux_, contravariant_vectors_,
                                        inverse_jacobian_);
    Kokkos::parallel_for(
        "viscous_interface_flux",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0], n_elems_[1], n_elems_[2]}),
        functor);
  }

  KOKKOS_INLINE_FUNCTION static std::size_t
  face_dof(std::size_t face, std::size_t node_i, std::size_t node_j) {
    return face * Basis::NNodes * Basis::NNodes +
           DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const {
    apply_face_x(ielem, jelem, kelem);
    apply_face_y(ielem, jelem, kelem);
    apply_face_z(ielem, jelem, kelem);
  }

  KOKKOS_INLINE_FUNCTION void apply_face_x(std::size_t ielem, std::size_t jelem,
                                           std::size_t kelem) const {
    const std::size_t left_i = neighbors(ielem, jelem, kelem, 0, 0);
    const std::size_t left_j = neighbors(ielem, jelem, kelem, 0, 1);
    const std::size_t left_k = neighbors(ielem, jelem, kelem, 0, 2);
    if (left_i == static_cast<std::size_t>(-1) ||
        left_j == static_cast<std::size_t>(-1) ||
        left_k == static_cast<std::size_t>(-1)) {
      return;
    }

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
        const std::size_t left_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            Basis::NNodes - 1, jnode, knode);
        const std::size_t right_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(0, jnode, knode);
        const auto normal_right =
            face_normal(ielem, jelem, kelem, right_dof, 0);
        add_divergence_flux(left_i, left_j, left_k, face_dof(1, jnode, knode),
                            left_dof, ielem, jelem, kelem,
                            face_dof(0, jnode, knode), right_dof, normal_right);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void apply_face_y(std::size_t ielem, std::size_t jelem,
                                           std::size_t kelem) const {
    const std::size_t bottom_i = neighbors(ielem, jelem, kelem, 1, 0);
    const std::size_t bottom_j = neighbors(ielem, jelem, kelem, 1, 1);
    const std::size_t bottom_k = neighbors(ielem, jelem, kelem, 1, 2);
    if (bottom_i == static_cast<std::size_t>(-1) ||
        bottom_j == static_cast<std::size_t>(-1) ||
        bottom_k == static_cast<std::size_t>(-1)) {
      return;
    }

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t bottom_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, Basis::NNodes - 1, knode);
        const std::size_t top_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, 0, knode);
        const auto normal_top = face_normal(ielem, jelem, kelem, top_dof, 1);
        add_divergence_flux(bottom_i, bottom_j, bottom_k,
                            face_dof(3, inode, knode), bottom_dof, ielem, jelem,
                            kelem, face_dof(2, inode, knode), top_dof,
                            normal_top);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void apply_face_z(std::size_t ielem, std::size_t jelem,
                                           std::size_t kelem) const {
    const std::size_t back_i = neighbors(ielem, jelem, kelem, 2, 0);
    const std::size_t back_j = neighbors(ielem, jelem, kelem, 2, 1);
    const std::size_t back_k = neighbors(ielem, jelem, kelem, 2, 2);
    if (back_i == static_cast<std::size_t>(-1) ||
        back_j == static_cast<std::size_t>(-1) ||
        back_k == static_cast<std::size_t>(-1)) {
      return;
    }

    for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t back_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, jnode, Basis::NNodes - 1);
        const std::size_t front_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, 0);
        const auto normal_front =
            face_normal(ielem, jelem, kelem, front_dof, 2);
        add_divergence_flux(back_i, back_j, back_k, face_dof(5, inode, jnode),
                            back_dof, ielem, jelem, kelem,
                            face_dof(4, inode, jnode), front_dof, normal_front);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION std::array<T, 3>
  face_normal(std::size_t ielem, std::size_t jelem, std::size_t kelem,
              std::size_t dof, std::size_t direction) const {
    const T sign =
        inverse_jacobian(ielem, jelem, kelem, dof) >= T{0} ? T{1} : T{-1};
    return {
        sign * contravariant_vectors(ielem, jelem, kelem, dof, direction, 0),
        sign * contravariant_vectors(ielem, jelem, kelem, dof, direction, 1),
        sign * contravariant_vectors(ielem, jelem, kelem, dof, direction, 2)};
  }

  KOKKOS_INLINE_FUNCTION T dot_viscous_flux(
      std::size_t ielem, std::size_t jelem, std::size_t kelem, std::size_t dof,
      std::size_t var, const std::array<T, 3>& normal) const {
    return viscous_flux(ielem, jelem, kelem, dof, var, 0) * normal[0] +
           viscous_flux(ielem, jelem, kelem, dof, var, 1) * normal[1] +
           viscous_flux(ielem, jelem, kelem, dof, var, 2) * normal[2];
  }

  KOKKOS_INLINE_FUNCTION void
  add_divergence_flux(std::size_t secondary_i, std::size_t secondary_j,
                      std::size_t secondary_k, std::size_t secondary_face_index,
                      std::size_t secondary_dof, std::size_t primary_i,
                      std::size_t primary_j, std::size_t primary_k,
                      std::size_t primary_face_index, std::size_t primary_dof,
                      const std::array<T, 3>& primary_normal) const {
    for (std::size_t var = 0; var < Equations::NVARS; ++var) {
      const T flux_primary = dot_viscous_flux(primary_i, primary_j, primary_k,
                                              primary_dof, var, primary_normal);
      const T flux_hat = flux_primary;
      surface_flux(primary_i, primary_j, primary_k, primary_face_index, var) =
          flux_hat;
      surface_flux(secondary_i, secondary_j, secondary_k, secondary_face_index,
                   var) = flux_hat;
    }
  }

  IndexArray neighbors;
  VectorFieldArray viscous_flux;
  DataArray surface_flux;
  MetricArray contravariant_vectors;
  ScalarArray inverse_jacobian;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct ParabolicJacobianProjFunctor;

template <class T, class Basis, class Equations>
struct ParabolicJacobianProjFunctor<T, Basis, Equations, 3> {
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using ScalarArray = typename scalar_node_type_traits<T, 3>::ScalarArray;

  ParabolicJacobianProjFunctor(DataArray du_, ScalarArray inverse_jacobian_)
      : du(du_), inverse_jacobian(inverse_jacobian_) {}

  static void apply(DataArray du_, ScalarArray inverse_jacobian_,
                    const std::array<std::size_t, 3>& n_elems_) {
    ParabolicJacobianProjFunctor functor(du_, inverse_jacobian_);
    Kokkos::parallel_for(
        "parabolic_jacobian_proj",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_elems_[0], n_elems_[1], n_elems_[2]}),
        functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const {
    constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes * Basis::NNodes;
    for (std::size_t dof = 0; dof < ndofs; ++dof) {
      const T factor = inverse_jacobian(ielem, jelem, kelem, dof);
      for (std::size_t var = 0; var < Equations::NVARS; ++var) {
        du(ielem, jelem, kelem, dof, var) *= factor;
      }
    }
  }

  DataArray du;
  ScalarArray inverse_jacobian;
};

} // namespace DGSEM
