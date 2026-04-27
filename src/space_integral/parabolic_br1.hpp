#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>

namespace DGSEM {

template <class Equations>
concept ParabolicEquations = requires(
    const Equations& eq,
    const std::array<typename equations::EquationTraits<Equations>::value_type,
                     equations::EquationTraits<Equations>::NVARS>& u,
    const std::array<
        std::array<typename equations::EquationTraits<Equations>::value_type,
                   Equations::NGRAD_VARS>,
        equations::EquationTraits<Equations>::NDIMS>& grad_q) {
  Equations::NGRAD_VARS;
  { eq.gradient_variables(u) };
  { eq.viscous_flux(eq.gradient_variables(u), grad_q, std::size_t{0}) };
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct GradientVolumeIntegralFunctor;

template <class T, class Basis, class Equations>
struct GradientVolumeIntegralFunctor<T, Basis, Equations, 3> {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using BasisData = typename Basis::DeviceData;

  GradientVolumeIntegralFunctor(DataArray u_, VectorFieldArray gradient_,
                                const Equations& eq_)
      : u(u_), gradient(gradient_), eq(eq_), basis(Basis::device_data()) {}

  static void apply(DataArray u_, VectorFieldArray gradient_,
                    const Equations& eq_,
                    const std::array<std::size_t, 3>& n_elems_) {
    GradientVolumeIntegralFunctor functor(u_, gradient_, eq_);
    Kokkos::parallel_for(
        "gradient_volume_integral",
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
          const auto q = load_gradient_variables(ielem, jelem, kelem, dof);

          for (std::size_t ii = 0; ii < Basis::NNodes; ++ii) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(ii, jnode, knode);
            for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
              gradient(ielem, jelem, kelem, out_dof, var, 0) +=
                  basis.derivative_dhat(ii, inode) * q[var];
            }
          }

          for (std::size_t jj = 0; jj < Basis::NNodes; ++jj) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jj, knode);
            for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
              gradient(ielem, jelem, kelem, out_dof, var, 1) +=
                  basis.derivative_dhat(jj, jnode) * q[var];
            }
          }

          for (std::size_t kk = 0; kk < Basis::NNodes; ++kk) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, kk);
            for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
              gradient(ielem, jelem, kelem, out_dof, var, 2) +=
                  basis.derivative_dhat(kk, knode) * q[var];
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION std::array<T, Equations::NGRAD_VARS>
  load_gradient_variables(std::size_t ielem, std::size_t jelem,
                          std::size_t kelem, std::size_t dof) const {
    std::array<T, traits::NVARS> state{};
    for (std::size_t var = 0; var < traits::NVARS; ++var) {
      state[var] = u(ielem, jelem, kelem, dof, var);
    }
    return eq.gradient_variables(state);
  }

  DataArray u;
  VectorFieldArray gradient;
  Equations eq;
  BasisData basis;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct GradientPhysicalTransformFunctor;

template <class T, class Basis, class Equations>
struct GradientPhysicalTransformFunctor<T, Basis, Equations, 3> {
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using MetricArray = typename jacobian_type_traits<T, 3>::JacobianMatrix;

  GradientPhysicalTransformFunctor(VectorFieldArray gradient_,
                                   MetricArray contravariant_vectors_)
      : gradient(gradient_), contravariant_vectors(contravariant_vectors_) {}

  static void apply(VectorFieldArray gradient_,
                    MetricArray contravariant_vectors_,
                    const std::array<std::size_t, 3>& n_elems_) {
    GradientPhysicalTransformFunctor functor(gradient_, contravariant_vectors_);
    Kokkos::parallel_for("gradient_transform",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {n_elems_[0] * Basis::NNodes,
                                         n_elems_[1] * Basis::NNodes,
                                         n_elems_[2] * Basis::NNodes}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         const std::size_t& kdof) const {
    // constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes *
    // Basis::NNodes;

    // for (std::size_t dof = 0; dof < ndofs; ++dof) {
    //   const T ja11 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 0);
    //   const T ja12 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 1);
    //   const T ja13 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 2);
    //   const T ja21 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 0);
    //   const T ja22 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 1);
    //   const T ja23 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 2);
    //   const T ja31 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 0);
    //   const T ja32 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 1);
    //   const T ja33 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 2);

    //   for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
    //     const T ref_xi = gradient(ielem, jelem, kelem, dof, var, 0);
    //     const T ref_eta = gradient(ielem, jelem, kelem, dof, var, 1);
    //     const T ref_zeta = gradient(ielem, jelem, kelem, dof, var, 2);

    //     gradient(ielem, jelem, kelem, dof, var, 0) =
    //         ja11 * ref_xi + ja21 * ref_eta + ja31 * ref_zeta;
    //     gradient(ielem, jelem, kelem, dof, var, 1) =
    //         ja12 * ref_xi + ja22 * ref_eta + ja32 * ref_zeta;
    //     gradient(ielem, jelem, kelem, dof, var, 2) =
    //         ja13 * ref_xi + ja23 * ref_eta + ja33 * ref_zeta;
    //   }
    // }

    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t kelem = kdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node_k = kdof % Basis::NNodes;

    const std::size_t dof =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j, node_k);

    const T ja11 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 0);
    const T ja12 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 1);
    const T ja13 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 2);
    const T ja21 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 0);
    const T ja22 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 1);
    const T ja23 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 2);
    const T ja31 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 0);
    const T ja32 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 1);
    const T ja33 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 2);

    for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
      const T ref_xi = gradient(ielem, jelem, kelem, dof, var, 0);
      const T ref_eta = gradient(ielem, jelem, kelem, dof, var, 1);
      const T ref_zeta = gradient(ielem, jelem, kelem, dof, var, 2);

      gradient(ielem, jelem, kelem, dof, var, 0) =
          ja11 * ref_xi + ja21 * ref_eta + ja31 * ref_zeta;
      gradient(ielem, jelem, kelem, dof, var, 1) =
          ja12 * ref_xi + ja22 * ref_eta + ja32 * ref_zeta;
      gradient(ielem, jelem, kelem, dof, var, 2) =
          ja13 * ref_xi + ja23 * ref_eta + ja33 * ref_zeta;
    }
  }

  // VectorFieldArray gradient_reference;
  VectorFieldArray gradient;
  MetricArray contravariant_vectors;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct GradientInterfaceFluxFunctor;

template <class T, class Basis, class Equations>
struct GradientInterfaceFluxFunctor<T, Basis, Equations, 3> {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using IndexArray = typename index_type_traits<3>::IndexArray;

  GradientInterfaceFluxFunctor(IndexArray neighbors_, const Equations& eq_,
                               DataArray u_, DataArray surface_flux_)
      : neighbors(neighbors_), eq(eq_), u(u_), surface_flux(surface_flux_) {}

  static void apply(IndexArray neighbors_, const Equations& eq_, DataArray u_,
                    DataArray surface_flux_,
                    const std::array<std::size_t, 3>& n_elems_) {
    GradientInterfaceFluxFunctor functor(neighbors_, eq_, u_, surface_flux_);
    Kokkos::parallel_for(
        "gradient_interface_flux",
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
        const auto q_left =
            load_gradient_variables(left_i, left_j, left_k, left_dof);
        const auto q_right =
            load_gradient_variables(ielem, jelem, kelem, right_dof);

        for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
          const T qhat = static_cast<T>(0.5) * (q_left[var] + q_right[var]);
          surface_flux(left_i, left_j, left_k, face_dof(1, jnode, knode), var) =
              qhat;
          surface_flux(ielem, jelem, kelem, face_dof(0, jnode, knode), var) =
              qhat;
        }
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
        const auto q_bottom =
            load_gradient_variables(bottom_i, bottom_j, bottom_k, bottom_dof);
        const auto q_top =
            load_gradient_variables(ielem, jelem, kelem, top_dof);

        for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
          const T qhat = static_cast<T>(0.5) * (q_bottom[var] + q_top[var]);
          surface_flux(bottom_i, bottom_j, bottom_k, face_dof(3, inode, knode),
                       var) = qhat;
          surface_flux(ielem, jelem, kelem, face_dof(2, inode, knode), var) =
              qhat;
        }
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
        const auto q_back =
            load_gradient_variables(back_i, back_j, back_k, back_dof);
        const auto q_front =
            load_gradient_variables(ielem, jelem, kelem, front_dof);

        for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
          const T qhat = static_cast<T>(0.5) * (q_back[var] + q_front[var]);
          surface_flux(back_i, back_j, back_k, face_dof(5, inode, jnode), var) =
              qhat;
          surface_flux(ielem, jelem, kelem, face_dof(4, inode, jnode), var) =
              qhat;
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION std::array<T, Equations::NGRAD_VARS>
  load_gradient_variables(std::size_t ielem, std::size_t jelem,
                          std::size_t kelem, std::size_t dof) const {
    std::array<T, traits::NVARS> state{};
    for (std::size_t var = 0; var < traits::NVARS; ++var) {
      state[var] = u(ielem, jelem, kelem, dof, var);
    }
    return eq.gradient_variables(state);
  }

  IndexArray neighbors;
  Equations eq;
  DataArray u;
  DataArray surface_flux;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct GradientSurfaceIntegralFunctor;

template <class T, class Basis, class Equations>
struct GradientSurfaceIntegralFunctor<T, Basis, Equations, 3> {
  using DataArray = typename solution_type_traits<T, 3>::DataArray;
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using MetricArray = typename jacobian_type_traits<T, 3>::JacobianMatrix;
  using ScalarArray = typename scalar_node_type_traits<T, 3>::ScalarArray;
  using BasisData = typename Basis::DeviceData;

  GradientSurfaceIntegralFunctor(VectorFieldArray gradient_,
                                 DataArray surface_flux_,
                                 MetricArray contravariant_vectors_,
                                 ScalarArray inverse_jacobian_)
      : gradient(gradient_), surface_flux(surface_flux_),
        contravariant_vectors(contravariant_vectors_),
        inverse_jacobian(inverse_jacobian_), basis(Basis::device_data()) {}

  static void apply(VectorFieldArray gradient_, DataArray surface_flux_,
                    MetricArray contravariant_vectors_,
                    ScalarArray inverse_jacobian_,
                    const std::array<std::size_t, 3>& n_elems_) {
    GradientSurfaceIntegralFunctor functor(
        gradient_, surface_flux_, contravariant_vectors_, inverse_jacobian_);
    Kokkos::parallel_for(
        "gradient_surface_integral",
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
    const T factor_1 = basis.boundary_interpolation_left[0];
    const T factor_2 = basis.boundary_interpolation_right[Basis::NNodes - 1];

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
        const std::size_t left_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(0, jnode, knode);
        const std::size_t right_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            Basis::NNodes - 1, jnode, knode);
        const auto left_normal = face_normal(ielem, jelem, kelem, left_dof, 0);
        const auto right_normal =
            face_normal(ielem, jelem, kelem, right_dof, 0);
        add_face_contribution(ielem, jelem, kelem, left_dof,
                              face_dof(0, jnode, knode), left_normal,
                              -factor_1);
        add_face_contribution(ielem, jelem, kelem, right_dof,
                              face_dof(1, jnode, knode), right_normal,
                              factor_2);
      }
    }

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t bottom_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, 0, knode);
        const std::size_t top_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, Basis::NNodes - 1, knode);
        const auto bottom_normal =
            face_normal(ielem, jelem, kelem, bottom_dof, 1);
        const auto top_normal = face_normal(ielem, jelem, kelem, top_dof, 1);
        add_face_contribution(ielem, jelem, kelem, bottom_dof,
                              face_dof(2, inode, knode), bottom_normal,
                              -factor_1);
        add_face_contribution(ielem, jelem, kelem, top_dof,
                              face_dof(3, inode, knode), top_normal, factor_2);
      }
    }

    for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t back_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, 0);
        const std::size_t front_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, jnode, Basis::NNodes - 1);
        const auto back_normal = face_normal(ielem, jelem, kelem, back_dof, 2);
        const auto front_normal =
            face_normal(ielem, jelem, kelem, front_dof, 2);
        add_face_contribution(ielem, jelem, kelem, back_dof,
                              face_dof(4, inode, jnode), back_normal,
                              -factor_1);
        add_face_contribution(ielem, jelem, kelem, front_dof,
                              face_dof(5, inode, jnode), front_normal,
                              factor_2);
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

  KOKKOS_INLINE_FUNCTION void
  add_face_contribution(std::size_t ielem, std::size_t jelem, std::size_t kelem,
                        std::size_t dof, std::size_t face_index,
                        const std::array<T, 3>& normal, T factor) const {
    for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
      const T qhat = surface_flux(ielem, jelem, kelem, face_index, var);
      for (std::size_t dim = 0; dim < 3; ++dim) {
        gradient(ielem, jelem, kelem, dof, var, dim) +=
            factor * qhat * normal[dim];
      }
    }
  }

  VectorFieldArray gradient;
  DataArray surface_flux;
  MetricArray contravariant_vectors;
  ScalarArray inverse_jacobian;
  BasisData basis;
};

template <class T, class Basis, class Equations, std::size_t NDIMS>
struct GradientJacobianProjFunctor;

template <class T, class Basis, class Equations>
struct GradientJacobianProjFunctor<T, Basis, Equations, 3> {
  using VectorFieldArray =
      typename node_vector_field_type_traits<T, 3>::DataArray;
  using ScalarArray = typename scalar_node_type_traits<T, 3>::ScalarArray;

  GradientJacobianProjFunctor(VectorFieldArray gradient_,
                              ScalarArray inverse_jacobian_)
      : gradient(gradient_), inverse_jacobian(inverse_jacobian_) {}

  static void apply(VectorFieldArray gradient_, ScalarArray inverse_jacobian_,
                    const std::array<std::size_t, 3>& n_elems_) {
    GradientJacobianProjFunctor functor(gradient_, inverse_jacobian_);
    Kokkos::parallel_for("gradient_jacobian_proj",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {n_elems_[0] * Basis::NNodes,
                                         n_elems_[1] * Basis::NNodes,
                                         n_elems_[2] * Basis::NNodes}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         const std::size_t& kdof) const {
    // constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes *
    // Basis::NNodes;

    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t kelem = kdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node_k = kdof % Basis::NNodes;
    const std::size_t dof =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j, node_k);

    // for (std::size_t dof = 0; dof < ndofs; ++dof) {
    const T factor = inverse_jacobian(ielem, jelem, kelem, dof);
    for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
      for (std::size_t dim = 0; dim < 3; ++dim) {
        gradient(ielem, jelem, kelem, dof, var, dim) *= factor;
      }
    }
    // }
  }

  VectorFieldArray gradient;
  ScalarArray inverse_jacobian;
};

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
    Kokkos::parallel_for("viscous_flux",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {n_elems_[0] * Basis::NNodes,
                                         n_elems_[1] * Basis::NNodes,
                                         n_elems_[2] * Basis::NNodes}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         const std::size_t& kdof) const {
    // constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes *
    // Basis::NNodes;
    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t kelem = kdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node_k = kdof % Basis::NNodes;
    const std::size_t dof =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j, node_k);

    // for (std::size_t dof = 0; dof < ndofs; ++dof) {
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

      const auto flux = eq.viscous_flux(q, grad_q, dim);
      for (std::size_t var = 0; var < traits::NVARS; ++var) {
        viscous_flux(ielem, jelem, kelem, dof, var, dim) = flux[var];
      }
    }

    // for (std::size_t dim = 0; dim < 3; ++dim) {
    //   const auto flux = eq.viscous_flux(q, grad_q, dim);
    //   for (std::size_t var = 0; var < traits::NVARS; ++var) {
    //     viscous_flux(ielem, jelem, kelem, dof, var, dim) = flux[var];
    //   }
    // }
    // }
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
    const std::array<T, 3> secondary_normal = {
        -primary_normal[0], -primary_normal[1], -primary_normal[2]};
    for (std::size_t var = 0; var < Equations::NVARS; ++var) {
      const T flux_primary = dot_viscous_flux(primary_i, primary_j, primary_k,
                                              primary_dof, var, primary_normal);
      // const T flux_secondary =
      //     dot_viscous_flux(secondary_i, secondary_j, secondary_k,
      //     secondary_dof,
      //                      var, secondary_normal);
      // const T flux_hat = static_cast<T>(0.5) * (flux_primary +
      // flux_secondary);
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
    Kokkos::parallel_for("parabolic_jacobian_proj",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {n_elems_[0] * Basis::NNodes,
                                         n_elems_[1] * Basis::NNodes,
                                         n_elems_[2] * Basis::NNodes}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& idof,
                                         const std::size_t& jdof,
                                         const std::size_t& kdof) const {
    // constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes *
    // Basis::NNodes;
    const std::size_t ielem = idof / Basis::NNodes;
    const std::size_t jelem = jdof / Basis::NNodes;
    const std::size_t kelem = kdof / Basis::NNodes;
    const std::size_t node_i = idof % Basis::NNodes;
    const std::size_t node_j = jdof % Basis::NNodes;
    const std::size_t node_k = kdof % Basis::NNodes;
    const std::size_t dof =
        DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j, node_k);
    // for (std::size_t dof = 0; dof < ndofs; ++dof) {
    const T factor = inverse_jacobian(ielem, jelem, kelem, dof);
    for (std::size_t var = 0; var < Equations::NVARS; ++var) {
      du(ielem, jelem, kelem, dof, var) *= factor;
    }
    // }
  }

  DataArray du;
  ScalarArray inverse_jacobian;
};

} // namespace DGSEM
