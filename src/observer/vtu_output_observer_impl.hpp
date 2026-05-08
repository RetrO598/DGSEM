#pragma once
#include <string>
#include <vector>

namespace DGSEM {
namespace detail {

void write_vtu_lagrange_quad(const std::string& filename,
                             const std::vector<double>& points,
                             const std::vector<double>& values,
                             const std::vector<std::string>& var_names, int nx,
                             int ny, int NNodes, int nvars);

void write_vtu_lagrange_hex(const std::string& filename,
                            const std::vector<double>& points,
                            const std::vector<double>& values,
                            const std::vector<std::string>& var_names, int nx,
                            int ny, int nz, int NNodes, int nvars);

} // namespace detail
} // namespace DGSEM
