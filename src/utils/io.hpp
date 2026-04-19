#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace DGSEM {
namespace utils {

// 写入 3D 解（device view 或 host view），期望视图形状为
// (n_elems_x, n_elems_y, n_elems_z, ndofs, nvars)
// 输出每一行：var inode jnode knode ielem jelem kelem value
template <class ViewType>
inline void write_solution_txt(const std::string& filename,
                               const ViewType& device_view, std::size_t NNodes,
                               const std::array<std::size_t, 3>& n_cells) {
  // 创建 host mirror 并拷贝
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_view);

  std::ofstream fout(filename);
  if (!fout) {
    std::cerr << "Failed to open file: " << filename << '\n';
    return;
  }

  const std::size_t nx = n_cells[0];
  const std::size_t ny = n_cells[1];
  const std::size_t nz = n_cells[2];

  const std::size_t ndofs = host_view.extent(3);
  const std::size_t nvars = host_view.extent(4);

  for (std::size_t ke = 0; ke < nz; ++ke) {
    for (std::size_t je = 0; je < ny; ++je) {
      for (std::size_t ie = 0; ie < nx; ++ie) {
        for (std::size_t dof = 0; dof < ndofs; ++dof) {
          for (std::size_t var = 0; var < nvars; ++var) {
            fout << std::setprecision(8) << std::scientific
                 << static_cast<double>(host_view(ie, je, ke, dof, var))
                 << '\n';
          }
        }
      }
    }
  }

  fout.close();
}

template <class ViewType>
inline void write_contravariant_txt(const std::string& filename,
                                    const ViewType& device_view,
                                    const std::array<std::size_t, 3>& n_cells) {
  // 创建 host mirror 并拷贝
  auto host_view =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_view);

  std::ofstream fout(filename);
  if (!fout) {
    std::cerr << "Failed to open file: " << filename << '\n';
    return;
  }

  const std::size_t nx = n_cells[0];
  const std::size_t ny = n_cells[1];
  const std::size_t nz = n_cells[2];

  const std::size_t ndofs = host_view.extent(3);

  for (std::size_t ke = 0; ke < nz; ++ke) {
    for (std::size_t je = 0; je < ny; ++je) {
      for (std::size_t ie = 0; ie < nx; ++ie) {
        for (std::size_t dof = 0; dof < ndofs; ++dof) {
          for (std::size_t var = 0; var < 3; ++var) {
            for (std::size_t var1 = 0; var1 < 3; ++var1) {
              fout << std::setprecision(8) << std::scientific
                   << static_cast<double>(host_view(ie, je, ke, dof, var, var1))
                   << '\n';
            }
          }
        }
      }
    }
  }

  fout.close();
}
} // namespace utils
} // namespace DGSEM
