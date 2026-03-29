#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <cstddef>

namespace DGSEM {
using Device = Kokkos::Device<Kokkos::DefaultExecutionSpace,
                              Kokkos::DefaultExecutionSpace::memory_space>;

template <class T, std::size_t NDIMS>
struct solution_type_traits {};

template <class T>
struct solution_type_traits<T, 1> {
  using DataArray = Kokkos::View<T***, Device>;
  using DataArrayHost = DataArray::HostMirror;
  using ElementArray = Kokkos::View<T*, Device>;
  using ElementArrayHost = ElementArray::HostMirror;
};

template <class T>
struct solution_type_traits<T, 2> {
  using DataArray = Kokkos::View<T****, Device>;
  using DataArrayHost = DataArray::HostMirror;
  using ElementArray = Kokkos::View<T**, Device>;
  using ElementArrayHost = ElementArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct coordinate_type_traits {};

template <class T>
struct coordinate_type_traits<T, 1> {
  using CoordArray = Kokkos::View<T***, Device>;
  using CoordArrayHost = CoordArray::HostMirror;
};

template <class T>
struct coordinate_type_traits<T, 2> {
  using CoordArray = Kokkos::View<T****, Device>;
  using CoordArrayHost = CoordArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct scalar_node_type_traits {};

template <class T>
struct scalar_node_type_traits<T, 1> {
  using ScalarArray = Kokkos::View<T**, Device>;
  using ScalarArrayHost = ScalarArray::HostMirror;
};

template <class T>
struct scalar_node_type_traits<T, 2> {
  using ScalarArray = Kokkos::View<T***, Device>;
  using ScalarArrayHost = ScalarArray::HostMirror;
};

template <std::size_t NDIMS>
struct index_type_traits {};

template <>
struct index_type_traits<1> {
  using IndexArray = Kokkos::View<std::size_t**, Device>;
  using IndexArrayHost = IndexArray::HostMirror;
};

template <>
struct index_type_traits<2> {
  using IndexArray = Kokkos::View<std::size_t****, Device>;
  using IndexArrayHost = IndexArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct jacobian_type_traits {};

template <class T>
struct jacobian_type_traits<T, 1> {
  using JacobianMatrix = Kokkos::View<T****, Device>;
  using JacobianMatrixHost = JacobianMatrix::HostMirror;
};

template <class T>
struct jacobian_type_traits<T, 2> {
  using JacobianMatrix = Kokkos::View<T*****, Device>;
  using JacobianMatrixHost = JacobianMatrix::HostMirror;
};
}  // namespace DGSEM
