#include "basis.hpp"
#include "data_container.hpp"
#include "equations.hpp"
#include "initial_condition.hpp"
#include "mapping.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "surface_flux.hpp"
#include "volume_flux.hpp"
#include <array>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
  using basis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;

  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    printf("Node %zu: % .16f, Weight: % .16f, Inv Weight: % .16f, Left Boundary"
           " Interpolation : % .16f, Right Boundary Interpolation: % .16f\n ",
           i, basis::nodes[i], basis::weights[i], basis::inv_weights[i],
           basis::boundary_interpolation_left[i],
           basis::boundary_interpolation_right[i]);
  }

  printf("Derivative matrix: \n");
  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    for (std::size_t j = 0; j < basis::NNodes; ++j) {
      printf(" % .16f", basis::derivative_matrix(i, j));
    }
    printf("\n");
  }

  printf("Derivative split matrix: \n");
  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    for (std::size_t j = 0; j < basis::NNodes; ++j) {
      printf(" % .16f", basis::derivative_split(i, j));
    }
    printf("\n");
  }

  printf("Derivative split transpose matrix: \n");
  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    for (std::size_t j = 0; j < basis::NNodes; ++j) {
      printf(" % .16f", basis::derivative_split_transpose(i, j));
    }
    printf("\n");
  }

  printf("Derivative dhat matrix: \n");
  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    for (std::size_t j = 0; j < basis::NNodes; ++j) {
      printf(" % .16f", basis::derivative_dhat(i, j));
    }
    printf("\n");
  }

  std::array<double, 1> left = {0.1};
  std::array<double, 1> right = {0.8};

  double left_scalar = 0.1;
  double right_scalar = 0.8;

  for (std::size_t i = 0; i < basis::NNodes; ++i) {
    std::cout << DGSEM::LinearMapping<double>::eval(basis::nodes[i],
                                                    left_scalar, right_scalar)
              << "\n";
  }

  for (auto &i : basis::nodes) {
    for (auto &j : basis::nodes) {
      auto res = DGSEM::LinearMapping<std::array<double, 2>>::eval(
          std::array<double, 2>{i, j}, std::array<double, 2>{0.0, 0.0},
          std::array<double, 2>{1.0, 1.0});
      std::cout << "(" << res[0] << ", " << res[1] << ")" << "\n";
    }
  }

  //   DGSEM::StructuredElementContainer<double, 1> container;

  //   DGSEM::StructuredElementInitializer<double, basis,
  //                                       DGSEM::LinearMapping<double>, 1>
  //       initializer{DGSEM::LinearMapping<double>(0.2, 2.85), {true}};

  //   initializer.init_elements({3}, container);
  //   std::cout
  //       << "=== Container: node_coordinates (shape={nelems,Nnodes,dim})
  //       ===\n";
  //   for (std::size_t e = 0; e < container.nelements[0]; ++e) {
  //     for (std::size_t n = 0; n < basis::NNodes; ++n) {
  //       std::cout << "  [e=" << e << ", n=" << n << "]: " <<
  //       std::setprecision(16)
  //                 << std::scientific << container.node_coordinates(e, n, 0)
  //                 << "\n";
  //     }
  //   }

  //   std::cout
  //       << "=== Container: jacobian_matrix (shape={nelems,Nnodes,1,1})
  //       ===\n";
  //   for (std::size_t e = 0; e < container.nelements[0]; ++e) {
  //     for (std::size_t n = 0; n < basis::NNodes; ++n) {
  //       std::cout << "  [e=" << e << ", n=" << n << "]: " <<
  //       std::setprecision(16)
  //                 << std::scientific << container.jacobian_matrix(e, n, 0, 0)
  //                 << "\n";
  //     }
  //   }

  //   std::cout << "=== Container: inverse_jacobian (shape={nelems,Nnodes})
  //   ===\n "; for (std::size_t e = 0; e < container.nelements[0]; ++e) {
  //       for (std::size_t n = 0; n < basis::NNodes; ++n) {
  //     std::cout << "  [e=" << e << ", n=" << n << "]: " <<
  //     std::setprecision(16)
  //               << std::scientific << container.inverse_jacobian(e, n) <<
  //               "\n";
  //   }
  // }

  // std::cout << "=== Container: contravariant_vectors "
  //              "(shape={nelems,Nnodes,1,1}) ===\n";
  // for (std::size_t e = 0; e < container.nelements[0]; ++e) {
  //   for (std::size_t n = 0; n < basis::NNodes; ++n) {
  //     std::cout << "  [e=" << e << ", n=" << n << "]: " <<
  //     std::setprecision(16)
  //               << std::scientific << container.contravariant_vectors(e, n,
  //               0, 0)
  //               << "\n";
  //   }
  // }

  // --- Test: verify Solver::initialize runs without errors ---
  using Eq = DGSEM::equations::LinearScalarAdvection1D<double>;
  using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;
  using VolumeFlux = DGSEM::VolumeIntegralWeakForm<MyBasis, Eq>;
  using SurfaceFlux = DGSEM::flux_godunov<Eq>;
  using Mesh = DGSEM::StructuredMesh<double, 1>;

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

  std::cout << "initial solution: " << "\n";
  for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
    for (std::size_t node = 0; node < basis::NNodes; ++node) {
      std::cout << sol.u(i, node, 0) << "\n";
    }
  }

  std::cout << "coordinates: " << "\n";
  for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
    for (std::size_t node = 0; node < basis::NNodes; ++node) {
      std::cout << container.node_coordinates(i, node, 0) << "\n";
    }
  }

  solver.calc_volume_integral(sol);

  std::cout << "volume integral: " << "\n";
  for (std::size_t node = 0; node < basis::NNodes; ++node) {
    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      std::cout << sol.du(i, node, 0) << "\n";
    }
  }

  std::ofstream file;
  file.open("volume.txt", std::ios::out);

  for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
    for (std::size_t node = 0; node < basis::NNodes; ++node) {
      file << std::scientific << std::setprecision(16) << sol.du(i, node, 0)
           << "\n";
    }
  }

  file.close();

  solver.calc_interface_flux(sol);

  std::cout << "interface: " << "\n";
  for (std::size_t node = 0; node < 2; ++node) {
    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      std::cout << sol.surface_flux_value(i, node, 0) << "\n";
    }
  }

  solver.calc_surface_integral(sol);

  solver.apply_jacobian(sol);

  std::cout << "du: " << "\n";
  for (std::size_t node = 0; node < basis::NNodes; ++node) {
    for (std::size_t i = 0; i < mesh.get_nelem(); ++i) {
      std::cout << sol.du(i, node, 0) << "\n";
    }
  }
}
