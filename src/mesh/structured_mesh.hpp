#pragma once
#include <Kokkos_Macros.hpp>
#include <array>
#include <boundary_condition/boundary_condition.hpp>
#include <cstddef>
#include <functional>
#include <numeric>

namespace DGSEM {
template <class T, std::size_t NDIM>
class StructuredMesh {
public:
  StructuredMesh(const std::array<T, NDIM>& domain_left_,
                 const std::array<T, NDIM>& domain_right_,
                 const std::array<std::size_t, NDIM> n_cells_)
      : n_cells(n_cells_), num_boundarys(2 * NDIM),
        nelem(std::accumulate(n_cells_.begin(), n_cells_.end(), std::size_t{1},
                              std::multiplies<std::size_t>())),
        domain_left(domain_left_), domain_right(domain_right_) {
    for (std::size_t i = 0; i < NDIM; ++i) {
      domain[2 * i] = domain_left[i];
      domain[2 * i + 1] = domain_right[i];
    }
  }

  StructuredMesh(const T& left, const T& right,
                 const std::array<std::size_t, NDIM> n_cells_)
    requires(NDIM == 1)
      : n_cells(n_cells_), num_boundarys(2 * NDIM),
        nelem(std::accumulate(n_cells_.begin(), n_cells_.end(), std::size_t{1},
                              std::multiplies<std::size_t>())) {
    domain[0] = left;
    domain[1] = right;
  }

  KOKKOS_INLINE_FUNCTION
  std::array<T, 2 * NDIM> get_domain_size() const { return domain; }

  KOKKOS_INLINE_FUNCTION
  std::array<std::size_t, NDIM> get_num_cells() const { return n_cells; }

  KOKKOS_INLINE_FUNCTION
  std::size_t get_num_cells(std::size_t i) const { return n_cells[i]; }

  KOKKOS_INLINE_FUNCTION
  std::size_t get_nelem() const { return nelem; }

  KOKKOS_INLINE_FUNCTION
  T get_face_coord(std::size_t face_id) const {
    if (face_id == 0) {
      return domain[0];
    } else if (face_id == 1) {
      return domain[1];
    }
    return T{};
  }

  KOKKOS_INLINE_FUNCTION
  std::array<T, NDIM> get_domain_left() const { return domain_left; }

  KOKKOS_INLINE_FUNCTION
  std::array<T, NDIM> get_domain_right() const { return domain_right; }

private:
  std::size_t num_boundarys;
  std::array<T, 2 * NDIM> domain; // [xmin, xmax, ymin, ymax, zmin, zmax]
  std::array<std::size_t, NDIM> n_cells;
  std::size_t nelem;
  std::array<T, NDIM> domain_left;
  std::array<T, NDIM> domain_right;
};

} // namespace DGSEM
