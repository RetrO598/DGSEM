#include "basis.hpp"
#include "data_container.hpp"
#include "equations.hpp"
#include "initial_condition.hpp"
#include "mapping.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "surface_flux.hpp"
#include "time_integrator.hpp"
#include "volume_flux.hpp"
#include <array>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {

  // --- Test: verify Solver::initialize runs without errors ---
  using Eq = DGSEM::equations::LinearScalarAdvection1D<double>;
  using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;
  using VolumeFlux = DGSEM::VolumeIntegralWeakForm<MyBasis, Eq>;
  using SurfaceFlux = DGSEM::flux_godunov<Eq>;
  using Mesh = DGSEM::StructuredMesh<double, 1>;
  using Solver =
      DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux, Mesh>;
  using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

  std::array<double, 2> domain_mesh = {-1.0, 1.0};
  std::array<std::size_t, 1> n_cells = {20};
  std::array<DGSEM::BoundaryCondition, 2> bcs = {
      DGSEM::BoundaryCondition::Periodic, DGSEM::BoundaryCondition::Periodic};

  Mesh mesh(domain_mesh, n_cells, bcs);
  Eq eq(1.0);

  DGSEM::StructuredElementContainer<double, 1> container;
  DGSEM::StructuredElementInitializer<double, MyBasis,
                                      DGSEM::LinearMapping<double>, 1>
      initializer{DGSEM::LinearMapping<double>(domain_mesh[0], domain_mesh[1]),
                  {true}};

  initializer.init_elements(n_cells, container);

  DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux, Mesh> solver(
      eq, mesh, container);

  DGSEM::Solution<Mesh, MyBasis, Eq> sol(mesh);

  DGSEM::SinwaveInitial<double> initial{};

  std::cout << "Testing solver.initialize()..." << std::endl;
  solver.initialize(initial, sol);
  std::cout << "solver.initialize() returned successfully." << std::endl;
  // solver.calc_rhs(sol);
  using TimeIntegrator = DGSEM::SSPRK3<double, Solver, Solution>;
  TimeIntegrator time_integrator(sol);
  const double t_final = 1.0; // One full period for wave speed c=1
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
    time_integrator.step(solver, sol, dt);
    t += dt;
    iter++;

    if (iter % 100 == 0) {
      std::cout << "Iter: " << std::setw(5) << iter << "  Time: " << std::fixed
                << std::setprecision(4) << t << std::endl;
    }
  }
  // solver.calc_rhs(sol);

  std::ofstream solution_file("solution.txt", std::ios::out);
  std::ofstream nodes_file("nodes.txt", std::ios::out);

  solution_file << std::scientific << std::setprecision(16);
  nodes_file << std::scientific << std::setprecision(16);

  for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
    for (std::size_t node = 0; node < MyBasis::NNodes; ++node) {
      solution_file << sol.u(i, node, 0) << "\n";
      nodes_file << container.node_coordinates(i, node, 0) << "\n";
    }
  }

  solution_file.close();
  nodes_file.close();
  std::cout << "Final solution saved to solution.txt and nodes.txt"
            << std::endl;

  return 0;
}

// #include "basis.hpp"
// #include "data_container.hpp"
// #include "equations.hpp"
// #include "initial_condition.hpp"
// #include "mapping.hpp"
// #include "mesh.hpp"
// #include "solver.hpp"
// #include "surface_flux.hpp"
// #include "time_integrator.hpp"
// #include "volume_flux.hpp"
// #include <array>
// #include <cstddef>
// #include <cstdio>
// #include <fstream>
// #include <iomanip>
// #include <iostream>

// int main() {
//   // --- 1. Set up problem parameters ---
//   using Eq = DGSEM::equations::LinearScalarAdvection1D<double>;
//   using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;
//   using VolumeFlux = DGSEM::VolumeIntegralWeakForm<MyBasis, Eq>;
//   using SurfaceFlux = DGSEM::flux_godunov<Eq>;
//   using Mesh = DGSEM::StructuredMesh<double, 1>;
//   using Solver =
//       DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux, Mesh>;
//   using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;
//   using TimeIntegrator = DGSEM::SSPRK3<double, Solver, Solution>;

//   const std::array<double, 2> domain = {-1.0, 1.0};
//   const std::array<std::size_t, 1> cells = {20};
//   const std::array<DGSEM::BoundaryCondition, 2> bcs = {
//       DGSEM::BoundaryCondition::Periodic,
//       DGSEM::BoundaryCondition::Periodic};
//   const double t_final = 1.0; // One full period for wave speed c=1
//   const double cfl = 0.1;

//   // --- 2. Create mesh, equation, and element data ---
//   Mesh mesh(domain, cells, bcs);
//   Eq eq(1.0); // Advection speed c=1.0

//   DGSEM::StructuredElementContainer<double, 1> element_container;
//   DGSEM::StructuredElementInitializer<double, MyBasis,
//                                       DGSEM::LinearMapping<double>, 1>
//       element_initializer{DGSEM::LinearMapping<double>(domain[0], domain[1]),
//                           {true}};
//   element_initializer.init_elements(cells, element_container);

//   // --- 3. Create solver and solution objects ---
//   Solver solver(eq, mesh, element_container);
//   Solution sol(mesh);

//   // --- 4. Set initial condition ---
//   DGSEM::SinwaveInitial<double> initial_condition{};
//   solver.initialize(initial_condition, sol);

//   // --- 5. Create time integrator and set time-stepping parameters ---
//   TimeIntegrator time_integrator(sol);

//   const double dx = (domain[1] - domain[0]) / cells[0];
//   const double dt = cfl * dx / eq.get_wave_speed();
//   double t = 0.0;
//   int iter = 0;

//   std::cout << "Starting simulation..." << std::endl;
//   std::cout << "  N_poly = " << MyBasis::NNodes - 1 << std::endl;
//   std::cout << "  N_elems = " << cells[0] << std::endl;
//   std::cout << "  dt = " << dt << std::endl;
//   std::cout << "  t_final = " << t_final << std::endl;

//   // --- 6. Main time loop ---
//   // while (t < t_final) {
//   //   time_integrator.step(solver, sol, dt);
//   //   t += dt;
//   //   iter++;

//   //   if (iter % 100 == 0) {
//   //     std::cout << "Iter: " << std::setw(5) << iter << "  Time: " <<
//   //     std::fixed
//   //               << std::setprecision(4) << t << std::endl;
//   //   }
//   // }

//   solver.calc_rhs(sol);

//   std::cout << "Simulation finished after " << iter << " iterations."
//             << std::endl;

//   // --- 7. Save final solution to file ---
//   std::ofstream solution_file("solution.txt", std::ios::out);
//   std::ofstream nodes_file("nodes.txt", std::ios::out);

//   solution_file << std::scientific << std::setprecision(16);
//   nodes_file << std::scientific << std::setprecision(16);

//   for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
//     for (std::size_t node = 0; node < MyBasis::NNodes; ++node) {
//       solution_file << sol.du(i, node, 0) << "\n";
//       nodes_file << element_container.node_coordinates(i, node, 0) << "\n";
//     }
//   }

//   solution_file.close();
//   nodes_file.close();
//   std::cout << "Final solution saved to solution.txt and nodes.txt"
//             << std::endl;

//   return 0;
// }
