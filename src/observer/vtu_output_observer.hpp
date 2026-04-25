#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <observer/observer_base.hpp>
#include <observer/vtu_output_observer_impl.hpp>
#include <string>
#include <utils/utils.hpp>
#include <vector>

namespace DGSEM {

template <class T, class Basis, class Solution, class Coord, class Equations>
class VTUOutputObserver : public SimulationObserver {
public:
  using trait = equations::EquationTraits<Equations>;
  constexpr static std::size_t NDIMS = trait::NDIMS;

  static constexpr int NNodes = Basis::NNodes; // = N+1
  static constexpr int N = NNodes - 1;
  static constexpr int NVARS = trait::NVARS;

  VTUOutputObserver(const std::string& output_path_, const Solution& sol_,
                    const Coord& coord_,
                    const std::array<std::size_t, NDIMS>& n_elems_,
                    int output_interval_ = -1)
      : output_path(output_path_), output_interval(output_interval_), sol(sol_),
        coord(coord_), n_elems(n_elems_) {}

  void on_step(int step, double time, bool&) override {

    if (step == 0) {
      write_vtu<Equations>(make_filename(step, time));
      last_written_step = step;
    }

    if (output_interval < 0) {
      return;
    }

    if (step % output_interval == 0) {
      write_vtu<Equations>(make_filename(step, time));
      last_written_step = step;
    }
  }

  void on_finish(int step, double time) override {
    if (step != last_written_step) {
      write_vtu<Equations>(make_filename(step, time));
    }
  }

  template <class equation>
  void write_vtu(const std::string& filename) const {
    if constexpr (NDIMS == 2) {
      write_vtu_2d<equation>(filename);
    } else if constexpr (NDIMS == 3) {
      write_vtu_3d<equation>(filename);
    }
  }

  template <class equation>
  void write_vtu_2d(const std::string& filename) const {
    std::size_t nx = n_elems[0];
    std::size_t ny = n_elems[1];

    const int ndof = NNodes * NNodes;
    const std::size_t nelem = nx * ny;
    const std::size_t total_points = nelem * ndof;

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);

    auto coord_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coord);

    constexpr bool is_euler =
        std::is_same_v<equation, equations::CompressibleEuler2D<T>>;

    const std::size_t nvars = is_euler ? NVARS + 1 : NVARS;

    std::vector<double> points(3 * total_points);
    std::vector<double> values(nvars * total_points);

    std::vector<std::string> var_names;

    if constexpr (is_euler) {
      var_names = {"rho", "rhou", "rhov", "rhoE", "pressure"};
    } else {
      for (std::size_t i = 0; i < NVARS; ++i) {
        var_names.push_back("var" + std::to_string(i));
      }
    }

    static auto tmp = vtkSmartPointer<vtkLagrangeQuadrilateral>::New();
    tmp->SetOrder(N, N);

    for (std::size_t je = 0; je < ny; ++je)
      for (std::size_t ie = 0; ie < nx; ++ie) {
        std::size_t e = je * nx + ie;

        for (int jd = 0; jd < NNodes; ++jd)
          for (int id = 0; id < NNodes; ++id) {
            int dg_id = jd * NNodes + id;

            int vtk_id = tmp->PointIndexFromIJK(id, jd, 0);
            std::size_t gid = e * ndof + vtk_id;

            // 坐标
            points[3 * gid + 0] = coord_host(ie, je, dg_id, 0);
            points[3 * gid + 1] = coord_host(ie, je, dg_id, 1);
            points[3 * gid + 2] = 0.0;

            std::array<T, NVARS> cons{};

            for (int v = 0; v < NVARS; ++v) {
              values[gid * nvars + v] = u_host(ie, je, dg_id, v);
              cons[v] = values[gid * nvars + v];
            }

            if constexpr (is_euler) {
              auto prim = DGSEM::utils::cons_to_prim(cons, 1.4);
              values[gid * nvars + NVARS] = prim[3]; // pressure
            }
          }
      }

    DGSEM::detail::write_vtu_lagrange_quad(filename, points, values, var_names,
                                           nx, ny, NNodes, nvars);
  }

  template <class equation>
  void write_vtu_3d(const std::string& filename) const {
    std::size_t nx = n_elems[0];
    std::size_t ny = n_elems[1];
    std::size_t nz = n_elems[2];

    const int ndof = NNodes * NNodes * NNodes;
    const std::size_t nelem = nx * ny * nz;
    const std::size_t total_points = nelem * ndof;

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);

    auto coord_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coord);

    constexpr bool is_euler =
        std::is_same_v<equation, equations::CompressibleEuler3D<T>>;

    constexpr bool is_navier_stokes =
        std::is_same_v<equation, equations::CompressibleNavierStokes3D<T>>;

    const std::size_t nvars =
        (is_euler or is_navier_stokes) ? NVARS + 1 : NVARS;

    std::vector<double> points(3 * total_points);
    std::vector<double> values(nvars * total_points);

    std::vector<std::string> var_names;

    if constexpr (is_euler) {
      var_names = {"rho", "rhou", "rhov", "rhow", "rhoE", "pressure"};
    } else if constexpr (is_navier_stokes) {
      var_names = {"rho", "rhou", "rhov", "rhow", "rhoE", "pressure"};
    } else {
      for (std::size_t i = 0; i < NVARS; ++i) {
        var_names.push_back("var" + std::to_string(i));
      }
    }

    static auto tmp = vtkSmartPointer<vtkLagrangeHexahedron>::New();
    tmp->SetOrder(N, N, N);

    for (std::size_t ke = 0; ke < nz; ++ke)
      for (std::size_t je = 0; je < ny; ++je)
        for (std::size_t ie = 0; ie < nx; ++ie) {
          std::size_t e = (ke * ny + je) * nx + ie;

          for (int kd = 0; kd < NNodes; ++kd)
            for (int jd = 0; jd < NNodes; ++jd)
              for (int id = 0; id < NNodes; ++id) {
                int dg_id = (kd * NNodes + jd) * NNodes + id;

                int vtk_id = tmp->PointIndexFromIJK(id, jd, kd);
                std::size_t gid = e * ndof + vtk_id;

                points[3 * gid + 0] = coord_host(ie, je, ke, dg_id, 0);
                points[3 * gid + 1] = coord_host(ie, je, ke, dg_id, 1);
                points[3 * gid + 2] = coord_host(ie, je, ke, dg_id, 2);

                std::array<T, NVARS> cons{};

                for (int v = 0; v < NVARS; ++v) {
                  values[gid * nvars + v] = u_host(ie, je, ke, dg_id, v);
                  cons[v] = values[gid * nvars + v];
                }

                if constexpr (is_euler or is_navier_stokes) {
                  auto prim = DGSEM::utils::cons_to_prim(cons, 1.4);
                  values[gid * nvars + NVARS] = prim[4]; // pressure
                }
              }
        }

    DGSEM::detail::write_vtu_lagrange_hex(filename, points, values, var_names,
                                          nx, ny, nz, NNodes, nvars);
  }

private:
  std::string make_filename(int step, double time) const {
    return output_path + "_step" + std::to_string(step) + ".vtu";
  }

private:
  const Solution& sol;
  const Coord& coord;

  std::string output_path;
  int output_interval = -1;
  int last_written_step = -1;

  const std::array<std::size_t, NDIMS>& n_elems;
};

} // namespace DGSEM