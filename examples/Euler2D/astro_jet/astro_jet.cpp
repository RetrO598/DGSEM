#include "Kokkos_Macros.hpp"
#include "boundary_condition/dirichlet_boundary.hpp"
#include "boundary_condition/periodic_boundary.hpp"
#include "utils/state_conversion.hpp"
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
struct AstroJetInitial
    : public DGSEM::AbstractInitial<AstroJetInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T rho = 0.5;
    T v1 = 0.0;
    T v2 = 0.0;
    T p = 0.4127;

    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p},
                                      static_cast<T>(1.4));
  }
};

template <class T>
struct AstroJetBoundaryState {
  KOKKOS_INLINE_FUNCTION std::array<T, 4>
  operator()(const std::array<T, 2>& coord, T time) const {
    T rho = 0.5;
    T v1 = 0.0;
    T v2 = 0.0;
    T p = 0.4127;
    if (time > 0.0 && std::abs(coord[0] + 0.5) < 1e-8 &&
        std::abs(coord[1]) < 0.05) {
      rho = 5.0;
      v1 = 800.0;
      v2 = 0.0;
      p = 0.4127;
    }

    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p}, gamma);
  }

  T gamma;
};

int main() {
  Kokkos::initialize();
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 3>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    using Mesh = DGSEM::StructuredMesh<value_type, 2>;

    MyBasis::initialize();

    AstroJetBoundaryState<value_type> inflow{1.4};

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::DirichletBC{inflow}, DGSEM::DirichletBC{inflow},
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 0.001;
    std::string output_path = "astro_jet.txt";

    std::array<value_type, 4> domain_mesh = {-0.5, 0.5, -0.5, 0.5};
    std::array<std::array<value_type, 2>, 2> mapping_domain = {
        std::array<value_type, 2>{-0.5, -0.5},
        std::array<value_type, 2>{0.5, 0.5}};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_mesh, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        mapping_domain[0], mapping_domain[1]),
                    {false, true}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.3, 0.0001, false);

    Solution sol(mesh);
    AstroJetInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh);

    const value_type cfl = 0.1;
    const value_type dx = (domain_mesh[1] - domain_mesh[0]) / nx;
    const value_type dy = (domain_mesh[3] - domain_mesh[2]) / ny;
    const value_type max_speed = 800.0;
    // const value_type dt =
    //     cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);
    const value_type dt = 1e-8;
    int iter = 0;
    value_type t = 0.0;

    using Analyzer = DGSEM::CompositeAnalyzer<
        MyBasis, Eq, DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;

    Analyzer analyzer;
    while (t < t_final) {
      time_integrator.step(solver, sol, dt);
      t += dt;
      iter++;

      DGSEM::AnalyzerFunctor<MyBasis, Eq, Analyzer, Solution>::apply(
          analyzer, sol, n_cells);

      if (analyzer.get<DGSEM::DivergenceChecker<value_type, Eq::NVARS>>()
              .has_nan) {
        std::cerr << "NaN detected at iter " << iter << ", time " << t
                  << std::endl;
        break;
      }

      if (iter % 100 == 0) {
        std::cout << "Iter: " << std::setw(5) << iter
                  << "  Time: " << std::fixed << std::setprecision(4) << t
                  << std::endl;
      }
    }

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
