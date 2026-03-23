#include "time_integrator/SSPRK3.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdio>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
  Kokkos::initialize();
  {
    using Eq = DGSEM::equations::LinearScalarAdvection1D<double>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;

    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;

    using VolumeFlux =
        DGSEM::VolumeIntegralShockCapturingHG<MyBasis, Eq, DGSEM::CentralFlux,
                                              DGSEM::LaxFriedrichsFlux,
                                              DGSEM::HGIndicator<MyBasis, Eq>>;

    auto boundaries =
        DGSEM::BoundarySet(DGSEM::PeriodicBC(), DGSEM::PeriodicBC());

    using Mesh = DGSEM::StructuredMesh<double, 1>;
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::array<double, 2> domain_mesh = {-1.0, 1.0};
    std::array<std::size_t, 1> n_cells = {120};
    // std::array<DGSEM::BoundaryCondition, 2> bcs = {
    //     DGSEM::BoundaryCondition::Periodic,
    //     DGSEM::BoundaryCondition::Periodic};

    Mesh mesh(domain_mesh, n_cells);
    Eq eq(1.0);

    DGSEM::StructuredElementContainer<double, 1> container;
    DGSEM::StructuredElementInitializer<double, MyBasis,
                                        DGSEM::LinearMapping<double>, 1>
        initializer{
            DGSEM::LinearMapping<double>(domain_mesh[0], domain_mesh[1]),
            {true}};

    initializer.init_elements(n_cells, container);

    container.sync_to_device();

    // container.check_data();

    Solver solver(eq, mesh, container, boundaries);

    DGSEM::Solution<Mesh, MyBasis, Eq> sol(mesh);

    DGSEM::CompositeWaveInitial<double> initial{};

    std::cout << "Testing solver.initialize()..." << std::endl;
    solver.initialize(initial, sol);
    std::cout << "solver.initialize() returned successfully." << std::endl;
    // sol.check_solution();
    // using TimeIntegrator = DGSEM::SSPRK3<double, Solver, Solution>;
    using TimeIntegrator = DGSEM::SSPRK3_kokkos<double, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh);
    const double t_final = 4.0;
    const double cfl = 0.1;
    const double dx = (domain_mesh[1] - domain_mesh[0]) / n_cells[0];
    const double dt = cfl * dx / eq.get_wave_speed();
    double t = 0.0;
    int iter = 0;

    std::cout << "Starting simulation..." << std::endl;
    std::cout << "  N_poly = " << MyBasis::NNodes - 1 << std::endl;
    std::cout << "  N_elems = " << n_cells[0] << std::endl;
    std::cout << "  dt = " << dt << std::endl;
    std::cout << "  t_final = " << t_final << std::endl;

    while (t < t_final) {
      time_integrator.step_kokkos(solver, sol, dt);
      t += dt;
      iter++;

      if (iter % 100 == 0) {
        std::cout << "Iter: " << std::setw(5) << iter
                  << "  Time: " << std::fixed << std::setprecision(4) << t
                  << std::endl;
      }
    }

    // time_integrator.step_kokkos(solver, sol, dt);

    // // solver.calc_rhs(sol);
    // sol.check_solution();
    // time_integrator.step_kokkos(solver, sol, dt);
    // sol.check_solution();

    std::ofstream solution_file("solution.txt", std::ios::out);
    std::ofstream solution_kokkos_file("solution_kokkos.txt", std::ios::out);

    std::ofstream nodes_file("nodes.txt", std::ios::out);

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_kokkos);

    solution_file << std::scientific << std::setprecision(16);
    solution_kokkos_file << std::scientific << std::setprecision(16);
    nodes_file << std::scientific << std::setprecision(16);

    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      for (std::size_t node = 0; node < MyBasis::NNodes; ++node) {
        solution_kokkos_file << u_host(i, node, 0) << "\n";
        // solution_file << sol.u(i, node, 0) << "\n";
        nodes_file << container.node_coordinates(i, node, 0) << "\n";
      }
    }

    solution_file.close();
    nodes_file.close();
    std::cout << "Final solution saved to solution.txt and nodes.txt"
              << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
