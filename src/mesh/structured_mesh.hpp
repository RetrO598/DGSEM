#pragma once
#include <array>
#include <boundary_condition/boundary_condition.hpp>
#include <cstddef>
#include <functional>
#include <numeric>
namespace DGSEM {
template <class T, std::size_t NDIM>
class StructuredMesh {
public:
  StructuredMesh(const std::array<T, 2 * NDIM> &domain_,
                 const std::array<std::size_t, NDIM> n_cells_,
                 const std::array<BoundaryCondition, 2 * NDIM> &bc)
      : domain(domain_), n_cells(n_cells_), boundary_conditions(bc),
        num_boundarys(2 * NDIM),
        nelem(std::accumulate(n_cells_.begin(), n_cells_.end(), 1.0,
                              std::multiplies<std::size_t>())) {}

  std::size_t get_num_boundarys() const { return num_boundarys; }

  BoundaryCondition get_boundary(std::size_t i) const {
    return boundary_conditions[i];
  }

  std::array<T, 2 * NDIM> get_domain_size() const { return domain; }

  std::array<std::size_t, NDIM> get_num_cells() const { return n_cells; }

  std::size_t get_num_cells(std::size_t i) const { return n_cells[i]; }

  std::size_t get_nelem() const { return nelem; }

private:
  std::size_t num_boundarys;
  std::array<T, 2 * NDIM> domain; // [xmin, xmax, ymin, ymax, zmin, zmax]
  std::array<std::size_t, NDIM> n_cells;
  std::array<BoundaryCondition, 2 * NDIM>
      boundary_conditions; // [xmin, ymin, zmin, xmax, ymax, zmax]
  std::size_t nelem;
};

} // namespace DGSEM
