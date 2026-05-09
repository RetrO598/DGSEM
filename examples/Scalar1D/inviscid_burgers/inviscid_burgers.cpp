#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>

int main() {
  DGSEM::KokkosSession kokkos;
  {
    using Eq = DGSEM::equations::InviscidBurgers1D<double>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 4>;

    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;

    using VolumeFlux =
        DGSEM::VolumeIntegralShockCapturingHG<MyBasis, Eq, DGSEM::CentralFlux,
                                              DGSEM::LaxFriedrichsFlux,
                                              DGSEM::HGIndicator<MyBasis, Eq>>;
    DGSEM::BasisGuard<MyBasis> basis;
    auto dirichFunc =
        KOKKOS_LAMBDA(const std::array<double, 1>& coordinate, double time) {
      double x = coordinate[0];
      double u = 0.0;
      u = std::sin(2.0 * std::numbers::pi * x);
      return std::array<double, 1>{u};
    };

    auto boundaries = DGSEM::BoundarySet(DGSEM::DirichletBC(dirichFunc),
                                         DGSEM::DirichletBC(dirichFunc));

    // std::array<double, 2> domain_mesh = {0.0, 1.0};
    std::array<double, 1> domain_left = {0.0};
    std::array<double, 1> domain_right = {1.0};
    std::array<std::size_t, 1> n_cells = {64};
    // std::array<DGSEM::BoundaryCondition, 2> bcs = {
    //     DGSEM::BoundaryCondition::Extrapolate,
    //     DGSEM::BoundaryCondition::Extrapolate};

    Eq eq{};
    auto problem =
        DGSEM::make_structured_problem<MyBasis, VolumeFlux, SurfaceFlux>(
            eq, domain_left, domain_right, n_cells, boundaries, {true});
    auto& solver = problem.solver();
    auto& sol = problem.solution();
    auto& container = problem.elements();
    const auto& mesh = problem.mesh();
    using Solution = typename decltype(problem)::SolutionType;

    DGSEM::BurgersSinwaveInitial<double> initial{};

    std::cout << "Testing solver.initialize()..." << std::endl;
    problem.initialize(initial);
    std::cout << "solver.initialize() returned successfully." << std::endl;

    auto time_integrator = problem.make_ssprk3();
    const double t_final = 2.0;
    const double cfl = 0.1;
    const double dx = (domain_right[0] - domain_left[0]) / n_cells[0];
    const double dt = cfl * dx / 1.0;
    double t = 0.0;
    int iter = 0;

    std::cout << "Starting simulation..." << std::endl;
    std::cout << "  N_poly = " << MyBasis::NNodes - 1 << std::endl;
    std::cout << "  N_elems = " << n_cells[0] << std::endl;
    std::cout << "  dt = " << dt << std::endl;
    std::cout << "  t_final = " << t_final << std::endl;

    while (t < t_final) {
      time_integrator.step(solver, sol, dt);
      t += dt;
      iter++;

      if (iter % 100 == 0) {
        std::cout << "Iter: " << std::setw(5) << iter
                  << "  Time: " << std::fixed << std::setprecision(4) << t
                  << std::endl;
      }
    }

    std::ofstream solution_file("solution.txt", std::ios::out);
    std::ofstream solution_kokkos_file("solution_kokkos.txt", std::ios::out);
    std::ofstream nodes_file("nodes.txt", std::ios::out);

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);
    solution_file << std::scientific << std::setprecision(16);
    solution_kokkos_file << std::scientific << std::setprecision(16);
    nodes_file << std::scientific << std::setprecision(16);

    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      for (std::size_t node = 0; node < MyBasis::NNodes; ++node) {
        solution_kokkos_file << u_host(i, node, 0) << "\n";
        nodes_file << container.node_coordinates(i, node, 0) << "\n";
      }
    }

    solution_file.close();
    nodes_file.close();
    std::cout << "Final solution saved to solution.txt and nodes.txt"
              << std::endl;
  }
  return 0;
}
