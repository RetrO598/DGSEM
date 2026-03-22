#pragma once
#include "../containers/data_container.hpp"
#include <array>
#include <cstddef>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <equations/equations.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class Basis, class Equations, class Element>
struct JacobianProj;

template <class Basis, class Equations, class T>
struct JacobianProj<Basis, Equations, StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, 1>::DataArray;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static void
  apply(const StructuredElementContainer<T, 1> &element, std::size_t ielem,
        xt::xarray<T> &du) {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      T factor = -element.inverse_jacobian(ielem, i, 0);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, i, var) *= factor;
      }
    }
  }
};

template <class Basis, class Equations, class Element, std::size_t NDIMS>
struct JacobianProjFunctor;

template <class Basis, class Equations, class Element, std::size_t NDIMS>
struct JacobianProjFunctor {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;
  using IndexArray = typename index_type_traits<NDIMS>::IndexArray;
  constexpr static std::size_t NVARS = traits::NVARS;

  JacobianProjFunctor(DataArray du_, DataArray inverse_jacobian_)
      : du(du_), inverse_jacobian(inverse_jacobian_) {}

  static void apply(DataArray du_, DataArray inverse_jacobian_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    JacobianProjFunctor functor(du_, inverse_jacobian_);

    Kokkos::parallel_for("jacobian_proj", n_elems_[0], functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t &ielem) const
    requires(NDIMS == 1)
  {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      T factor = -inverse_jacobian(ielem, i, 0);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, i, var) *= factor;
      }
    }
  }

  DataArray du;
  DataArray inverse_jacobian;
};
} // namespace DGSEM