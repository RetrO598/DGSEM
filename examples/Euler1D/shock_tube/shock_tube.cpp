#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
  Kokkos::initialize();
  {
    using Eq = DGSEM::equations::CompressibleEuler1D<double>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 4>;

    using SurfaceFlux = DGSEM::HLLCFlux<Eq>;

    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::HLLCFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    MyBasis::initialize();
    auto dirichFunc =
        KOKKOS_LAMBDA(const std::array<double, 1>& coordinate, double time) {
      double x = coordinate[0];
      double rho, u, p;
      if (x < 0.5) {
        rho = 1.0;
        u = 0.0;
        p = 1.0;
      } else {
        rho = 0.125;
        u = 0.0;
        p = 0.1;
      }
      return DGSEM::utils::prim_to_cons(std::array<double, 3>{rho, u, p}, 1.4);
    };

    auto boundaries = DGSEM::BoundarySet(DGSEM::DirichletBC(dirichFunc),
                                         DGSEM::DirichletBC(dirichFunc));
    using Mesh = DGSEM::StructuredMesh<double, 1>;
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    // std::array<double, 2> domain_mesh = {0.0, 1.0};
    std::array<double, 1> domain_left = {0.0};
    std::array<double, 1> domain_right = {1.0};

    std::array<std::size_t, 1> n_cells = {500};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq(1.4);

    DGSEM::StructuredElementContainer<double, 1> container;
    DGSEM::StructuredElementInitializer<double, MyBasis,
                                        DGSEM::LinearMapping<double>, 1>
        initializer{
            DGSEM::LinearMapping<double>(domain_left[0], domain_right[0]),
            {true}};

    initializer.init_elements(n_cells, container);

    // container.sync_to_device();

    Solver solver(eq, mesh, container, boundaries);

    DGSEM::Solution<Mesh, MyBasis, Eq> sol(mesh);

    DGSEM::SodShockTubeInitial<double> initial{};

    std::cout << "Testing solver.initialize()..." << std::endl;
    solver.initialize(initial, sol);
    std::cout << "solver.initialize() returned successfully." << std::endl;

    using TimeIntegrator = DGSEM::SSPRK3<double, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh);
    const double t_final = 0.2;
    const double cfl = 0.1;
    const double dx = (domain_right[0] - domain_left[0]) / n_cells[0];
    const double dt = cfl * dx / (1.0 + std::sqrt(1.4));
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

    std::ofstream solution_kokkos_file("solution_kokkos.txt", std::ios::out);
    std::ofstream nodes_file("nodes.txt", std::ios::out);

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);

    solution_kokkos_file << std::scientific << std::setprecision(16);
    nodes_file << std::scientific << std::setprecision(16);

    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      for (std::size_t node = 0; node < MyBasis::NNodes; ++node) {
        solution_kokkos_file << u_host(i, node, 0) << "\n";
        nodes_file << container.node_coordinates(i, node, 0) << "\n";
      }
    }

    solution_kokkos_file.close();
    nodes_file.close();
    std::cout << "Final solution saved to solution.txt and nodes.txt"
              << std::endl;

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
