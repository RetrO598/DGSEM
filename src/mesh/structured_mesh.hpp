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
  StructuredMesh(const std::array<T, 2 * NDIM>& domain_,
                 const std::array<std::size_t, NDIM> n_cells_)
      : domain(domain_), n_cells(n_cells_), num_boundarys(2 * NDIM),
        nelem(std::accumulate(n_cells_.begin(), n_cells_.end(), 1.0,
                              std::multiplies<std::size_t>())) {}
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

private:
  std::size_t num_boundarys;
  std::array<T, 2 * NDIM> domain; // [xmin, xmax, ymin, ymax, zmin, zmax]
  std::array<std::size_t, NDIM> n_cells;
  std::size_t nelem;
};

} // namespace DGSEM
