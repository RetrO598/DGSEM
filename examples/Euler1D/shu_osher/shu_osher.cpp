#include "base/numerical_flux.hpp"
#include "boundary_condition/initial_condition_base.hpp"
#include "equations/compressible_euler1D.hpp"
#include "space_integral/volume_flux.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {

  using Eq = DGSEM::equations::CompressibleEuler1D<double>;
  using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 4>;

  using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;

  using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
      MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
      DGSEM::HGIndicator<MyBasis, Eq>>;

  auto dirichFunc = [](const std::array<double, 1> &coordinate, double time) {
    double x = coordinate[0];
    double rho, u, p;
    if (x < -4.0) {
      rho = 3.857;
      u = 2.629;
      p = 10.333;
    } else {
      rho = 1.0 + 0.2 * std::sin(5.0 * x);
      u = 0.0;
      p = 1.0;
    }
    double mom = rho * u;
    double gamma = 1.4;
    double rhoE = p / (gamma - 1.0) + 0.5 * rho * u * u;
    return std::array<double, 3>{rho, mom, rhoE};
  };

  auto boundaries = DGSEM::BoundarySet(DGSEM::DirichletBC(dirichFunc),
                                       DGSEM::DirichletBC(dirichFunc));

  using Mesh = DGSEM::StructuredMesh<double, 1>;
  using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                         Mesh, decltype(boundaries)>;
  using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

  std::array<double, 2> domain_mesh = {-5.0, 5.0};
  std::array<std::size_t, 1> n_cells = {256};
  std::array<DGSEM::BoundaryCondition, 2> bcs = {
      DGSEM::BoundaryCondition::Extrapolate,
      DGSEM::BoundaryCondition::Extrapolate};

  Mesh mesh(domain_mesh, n_cells, bcs);
  Eq eq(1.4);

  DGSEM::StructuredElementContainer<double, 1> container;
  DGSEM::StructuredElementInitializer<double, MyBasis,
                                      DGSEM::LinearMapping<double>, 1>
      initializer{DGSEM::LinearMapping<double>(domain_mesh[0], domain_mesh[1]),
                  {false}};

  initializer.init_elements(n_cells, container);

  Solver solver(eq, mesh, container, boundaries);

  DGSEM::Solution<Mesh, MyBasis, Eq> sol(mesh);

  DGSEM::ShuOsherInitial<double> initial{};

  std::cout << "Testing solver.initialize()..." << std::endl;
  solver.initialize(initial, sol);
  std::cout << "solver.initialize() returned successfully." << std::endl;

  using TimeIntegrator = DGSEM::SSPRK3<double, Solver, Solution>;
  TimeIntegrator time_integrator(sol);
  const double t_final = 1.8;
  const double cfl = 0.05;
  const double dx = (domain_mesh[1] - domain_mesh[0]) / n_cells[0];
  const double dt = cfl * dx / 4.566;
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
