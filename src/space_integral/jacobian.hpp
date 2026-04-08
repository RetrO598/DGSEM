#pragma once

#include <array>
#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>

namespace DGSEM {
template <class Basis, class Equations, class Element>
struct JacobianProj;

template <class Basis, class Equations, class T>
struct JacobianProj<Basis, Equations, StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, 1>::DataArray;
  constexpr static std::size_t NVARS = traits::NVARS;
};

template <class Basis, class Equations, class Element, std::size_t NDIMS>
struct JacobianProjFunctor;

template <class Basis, class Equations, class Element, std::size_t NDIMS>
struct JacobianProjFunctor {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;
  using ScalarArray = typename scalar_node_type_traits<T, NDIMS>::ScalarArray;
  constexpr static std::size_t NVARS = traits::NVARS;

  JacobianProjFunctor(DataArray du_, ScalarArray inverse_jacobian_)
      : du(du_), inverse_jacobian(inverse_jacobian_) {}

  static void apply(DataArray du_, ScalarArray inverse_jacobian_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    JacobianProjFunctor functor(du_, inverse_jacobian_);

    Kokkos::parallel_for("jacobian_proj", n_elems_[0], functor);
  }

  static void apply(DataArray du_, ScalarArray inverse_jacobian_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 2)
  {
    JacobianProjFunctor functor(du_, inverse_jacobian_);
    Kokkos::parallel_for("jacobian_proj",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_elems_[0], n_elems_[1]}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem) const
    requires(NDIMS == 1)
  {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      T factor = -inverse_jacobian(ielem, i);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, i, var) *= factor;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem) const
    requires(NDIMS == 2)
  {
    constexpr std::size_t ndofs = Basis::NNodes * Basis::NNodes;
    for (std::size_t dof = 0; dof < ndofs; ++dof) {
      const T factor = -inverse_jacobian(ielem, jelem, dof);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, jelem, dof, var) *= factor;
      }
    }
  }

  DataArray du;
  ScalarArray inverse_jacobian;
};
} // namespace DGSEM
