#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

template <class T>
struct Riemann2DInitial
    : public DGSEM::AbstractInitial<Riemann2DInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    const T x = coordinate[0];
    const T y = coordinate[1];

    T rho;
    T v1;
    T v2;
    T p;
    if (x >= 0.5 && y >= 0.5) {
      rho = 0.5313;
      v1 = 0.0;
      v2 = 0.0;
      p = 0.4;
    } else if (x < 0.5 && y >= 0.5) {
      rho = 1.0;
      v1 = 0.7276;
      v2 = 0.0;
      p = 1.0;
    } else if (x < 0.5 && y < 0.5) {
      rho = 0.8;
      v1 = 0.0;
      v2 = 0.0;
      p = 1.0;
    } else if (x >= 0.5 && y < 0.5) {
      rho = 1.0;
      v1 = 0.0;
      v2 = 0.7276;
      p = 1.0;
    }

    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  Kokkos::initialize();
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 4>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    using Mesh = DGSEM::StructuredMesh<value_type, 2>;

    MyBasis::initialize();

    auto boundaries =
        DGSEM::BoundarySet(DGSEM::OutflowBC{}, DGSEM::OutflowBC{},
                           DGSEM::OutflowBC{}, DGSEM::OutflowBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 0.25;
    std::string output_path = "riemann2d.txt";

    std::array<value_type, 4> domain_mesh = {0.0, 1.0, 0.0, 1.0};
    std::array<std::array<value_type, 2>, 2> mapping_domain = {
        std::array<value_type, 2>{0.0, 0.0},
        std::array<value_type, 2>{1.0, 1.0}};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_mesh, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        mapping_domain[0], mapping_domain[1]),
                    {false, false}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.5, 0.001, false);

    Solution sol(mesh);
    Riemann2DInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using Analyzer = DGSEM::CompositeAnalyzer<
        MyBasis, Eq, DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using AnalyzerObserver =
        DGSEM::AnalyzerObserver<MyBasis, Eq, Solution, Analyzer>;

    using DGSEM::PrintObserver;

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh, t_final);

    // CFL, dt 计算
    const value_type cfl = 0.2;
    const value_type dx = (domain_mesh[1] - domain_mesh[0]) / nx;
    const value_type dy = (domain_mesh[3] - domain_mesh[2]) / ny;
    const value_type max_speed = 2.0;
    const value_type dt =
        cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);

    // 注册 observer
    time_integrator.add_observer(
        std::make_unique<AnalyzerObserver>(analyzer, sol, n_cells));
    time_integrator.add_observer(std::make_unique<PrintObserver>(100));

    // 用 observer 驱动积分
    time_integrator.solve(solver, sol, dt);

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);
    auto coord_host = container.node_coordinates;

    std::ofstream solution_file(output_path, std::ios::out);
    solution_file << std::scientific << std::setprecision(16);
    const std::size_t ndofs = MyBasis::NNodes * MyBasis::NNodes;

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t dof = 0; dof < ndofs; ++dof) {
          const std::array<value_type, 4> u_cons = {
              u_host(ielem, jelem, dof, 0), u_host(ielem, jelem, dof, 1),
              u_host(ielem, jelem, dof, 2), u_host(ielem, jelem, dof, 3)};
          const auto u_prim = DGSEM::utils::cons_to_prim(u_cons, eq);
          solution_file << coord_host(ielem, jelem, dof, 0) << " "
                        << coord_host(ielem, jelem, dof, 1) << " " << u_prim[0]
                        << " " << u_prim[1] << " " << u_prim[2] << " "
                        << u_prim[3] << "\n";
        }
      }
    }

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
