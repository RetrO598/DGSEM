#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <equations/equations.hpp>
#include <observer/observer_base.hpp>
#include <observer/vtu_output_observer_impl.hpp>
#include <string>
#include <vector>

namespace DGSEM {

template <class T, class Basis, class Solution, class Coord, class Equations>
class VTUOutputObserver : public SimulationObserver {};

template <class T, class Basis, class Solution, class Coord>
class VTUOutputObserver<T, Basis, Solution, Coord,
                        equations::CompressibleEuler2D<T>>
    : public SimulationObserver {
public:
  using Eq = equations::CompressibleEuler2D<T>;
  using trait = equations::EquationTraits<Eq>;
  constexpr static std::size_t NDIMS = trait::NDIMS;

  static constexpr int NNodes = Basis::NNodes; // = N+1
  static constexpr int N = NNodes - 1;
  static constexpr int NVARS = Eq::NVARS;

  VTUOutputObserver(const std::string& output_path_, const Solution& sol_,
                    const Coord& coord_,
                    const std::array<std::size_t, NDIMS>& n_elems_,
                    int output_interval_ = -1)
      : output_path(output_path_), output_interval(output_interval_), sol(sol_),
        coord(coord_), n_elems(n_elems_) {}

  void on_step(int step, double time, bool&) override {

    if (step == 0) {
      write_vtu(make_filename(step, time));
      last_written_step = step;
    }

    if (output_interval < 0) {
      return;
    }

    if (step % output_interval == 0) {
      write_vtu(make_filename(step, time));
      last_written_step = step;
    }
  }

  void on_finish(int step, double time) override {
    if (step != last_written_step) {
      write_vtu(make_filename(step, time));
    }
  }

private:
  std::string make_filename(int step, double time) const {
    return output_path + "_step" + std::to_string(step) + ".vtu";
  }

  void write_vtu(const std::string& filename) const {
    std::size_t nx = n_elems[0];
    std::size_t ny = n_elems[1];

    const int ndof = NNodes * NNodes;
    const std::size_t nelem = nx * ny;
    const std::size_t total_points = nelem * ndof;

    // ----------------------------
    // host copy
    // ----------------------------
    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);

    auto coord_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coord);

    // ----------------------------
    // 构造点（每个element独立）
    // ----------------------------
    std::vector<double> points(3 * total_points);
    std::vector<double> values(NVARS * total_points);

    for (std::size_t je = 0; je < ny; ++je)
      for (std::size_t ie = 0; ie < nx; ++ie) {
        std::size_t e = je * nx + ie;

        for (int jd = 0; jd < NNodes; ++jd)
          for (int id = 0; id < NNodes; ++id) {
            int dg_id = jd * NNodes + id;
            static auto tmp = vtkSmartPointer<vtkLagrangeQuadrilateral>::New();
            tmp->SetOrder(N, N);

            int vtk_id = tmp->PointIndexFromIJK(id, jd, 0);
            std::size_t gid = e * ndof + vtk_id;

            // 坐标
            points[3 * gid + 0] = coord_host(ie, je, dg_id, 0);
            points[3 * gid + 1] = coord_host(ie, je, dg_id, 1);
            points[3 * gid + 2] = 0.0;

            // 变量
            for (int v = 0; v < NVARS; ++v)
              values[gid * NVARS + v] = u_host(ie, je, dg_id, v);
          }
      }

    DGSEM::detail::write_vtu_lagrange_quad(filename, points, values, nx, ny,
                                           NNodes, NVARS);
  }

private:
  const Solution& sol;
  const Coord& coord;

  std::string output_path;
  int output_interval = -1;
  int last_written_step = -1;

  const std::array<std::size_t, NDIMS>& n_elems;
};

template <class T, class Basis, class Solution, class Coord>
class VTUOutputObserver<T, Basis, Solution, Coord,
                        equations::CompressibleEuler3D<T>>
    : public SimulationObserver {
public:
  using Eq = equations::CompressibleEuler3D<T>;
  using trait = equations::EquationTraits<Eq>;
  constexpr static std::size_t NDIMS = trait::NDIMS;

  static constexpr int NNodes = Basis::NNodes; // = N+1
  static constexpr int N = NNodes - 1;
  static constexpr int NVARS = Eq::NVARS;

  VTUOutputObserver(const std::string& output_path_, const Solution& sol_,
                    const Coord& coord_,
                    const std::array<std::size_t, NDIMS>& n_elems_,
                    int output_interval_ = -1)
      : output_path(output_path_), output_interval(output_interval_), sol(sol_),
        coord(coord_), n_elems(n_elems_) {}

  void on_step(int step, double time, bool&) override {
    if (step == 0) {
      write_vtu(make_filename(step, time));
      last_written_step = step;
    }

    if (output_interval < 0) {
      return;
    }

    if (step % output_interval == 0) {
      write_vtu(make_filename(step, time));
      last_written_step = step;
    }
  }

  void on_finish(int step, double time) override {
    if (step != last_written_step) {
      write_vtu(make_filename(step, time));
    }
  }

private:
  std::string make_filename(int step, double time) const {
    return output_path + "_step" + std::to_string(step) + ".vtu";
  }

  void write_vtu(const std::string& filename) const {
    std::size_t nx = n_elems[0];
    std::size_t ny = n_elems[1];
    std::size_t nz = n_elems[2];

    const int ndof = NNodes * NNodes * NNodes;
    const std::size_t nelem = nx * ny * nz;
    const std::size_t total_points = nelem * ndof;

    // ----------------------------
    // host copy
    // ----------------------------
    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);

    auto coord_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coord);

    // ----------------------------
    // 构造点（每个element独立）
    // ----------------------------
    std::vector<double> points(3 * total_points);
    std::vector<double> values(NVARS * total_points);

    for (std::size_t ke = 0; ke < nz; ++ke)
      for (std::size_t je = 0; je < ny; ++je)
        for (std::size_t ie = 0; ie < nx; ++ie) {
          std::size_t e = (ke * ny + je) * nx + ie;

          for (int kd = 0; kd < NNodes; ++kd)
            for (int jd = 0; jd < NNodes; ++jd)
              for (int id = 0; id < NNodes; ++id) {
                int dg_id = (kd * NNodes + jd) * NNodes + id;
                static auto tmp = vtkSmartPointer<vtkLagrangeHexahedron>::New();
                tmp->SetOrder(N, N, N);

                int vtk_id = tmp->PointIndexFromIJK(id, jd, kd);
                std::size_t gid = e * ndof + vtk_id;

                // 坐标
                points[3 * gid + 0] = coord_host(ie, je, ke, dg_id, 0);
                points[3 * gid + 1] = coord_host(ie, je, ke, dg_id, 1);
                points[3 * gid + 2] = coord_host(ie, je, ke, dg_id, 2);

                // 变量
                for (int v = 0; v < NVARS; ++v)
                  values[gid * NVARS + v] = u_host(ie, je, ke, dg_id, v);
              }
        }

    DGSEM::detail::write_vtu_lagrange_hex(filename, points, values, nx, ny, nz,
                                          NNodes, NVARS);
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