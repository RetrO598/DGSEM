#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cstddef>
#include <decl/Kokkos_Declare_OPENMP.hpp>

namespace DGSEM {
using Device = Kokkos::Device<Kokkos::DefaultExecutionSpace,
                              Kokkos::DefaultExecutionSpace::memory_space>;

template <class T, std::size_t NDIMS>
struct solution_type_traits {};

template <class T>
struct solution_type_traits<T, 1> {
  using DataArray = Kokkos::View<T ***, Device>;
  using DataArrayHost = DataArray::HostMirror;
};

template <std::size_t NDIMS>
struct index_type_traits {};

template <>
struct index_type_traits<1> {
  using IndexArray = Kokkos::View<std::size_t **, Device>;
};

template <class T, std::size_t NDIMS>
struct jacobian_type_traits {};

template <class T>
struct jacobian_type_traits<T, 1> {
  using JacobianMatrix = Kokkos::View<T ****, Device>;
};
} // namespace DGSEM
