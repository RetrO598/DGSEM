#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>

template <class T>
struct KelvinHelmholtzInitial
    : public DGSEM::AbstractInitial<KelvinHelmholtzInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    const T x = coordinate[0];
    const T y = coordinate[1];
    const T slope = static_cast<T>(15.0);
    const T B = std::tanh(slope * y + static_cast<T>(7.5)) -
                std::tanh(slope * y - static_cast<T>(7.5));
    const T rho = static_cast<T>(0.5) + static_cast<T>(0.75) * B;
    const T v1 = static_cast<T>(0.5) * (B - static_cast<T>(1.0));
    const T v2 = static_cast<T>(0.1) *
                 std::sin(static_cast<T>(2.0) * std::numbers::pi_v<T> * x);
    const T p = static_cast<T>(1.0);
    const T gamma = static_cast<T>(1.4);
    const T rhoE = p / (gamma - static_cast<T>(1.0)) +
                   static_cast<T>(0.5) * rho * (v1 * v1 + v2 * v2);
    return {rho, rho * v1, rho * v2, rhoE};
  }
};

template <class T>
std::array<T, 4> cons_to_prim(const std::array<T, 4>& u, T gamma) {
  const T rho = u[0];
  const T rhou = u[1];
  const T rhov = u[2];
  const T rhoE = u[3];
  const T v1 = rhou / rho;
  const T v2 = rhov / rho;
  const T p = (gamma - static_cast<T>(1.0)) *
              (rhoE - static_cast<T>(0.5) * rho * (v1 * v1 + v2 * v2));
  return {rho, v1, v2, p};
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
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
        DGSEM::BoundarySet(DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
                           DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 3.7;
    std::string output_path = "kelvin_helmholtz_hg.txt";
    if (argc > 1)
      nx = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2)
      ny = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (argc > 3) t_final = std::strtod(argv[3], nullptr);
    if (argc > 4) output_path = argv[4];

    std::array<value_type, 4> domain_mesh = {-1.0, 1.0, -1.0, 1.0};
    std::array<std::array<value_type, 2>, 2> mapping_domain = {
        std::array<value_type, 2>{-1.0, -1.0},
        std::array<value_type, 2>{1.0, 1.0}};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_mesh, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        mapping_domain[0], mapping_domain[1]),
                    {true, true}};

    initializer.init_elements(n_cells, container);
    // container.sync_to_device();

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.002, 0.0001, false);

    Solution sol(mesh);
    KelvinHelmholtzInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh);

    const value_type cfl = 0.2;
    const value_type dx = (domain_mesh[1] - domain_mesh[0]) / nx;
    const value_type dy = (domain_mesh[3] - domain_mesh[2]) / ny;
    const value_type max_speed = 2.5;
    const value_type dt =
        cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);

    int iter = 0;
    value_type t = 0.0;
    while (t < t_final) {
      const value_type dt_step = std::min(dt, t_final - t);
      time_integrator.step(solver, sol, dt_step);
      t += dt_step;
      ++iter;

      if (iter % 100 == 0) {
        std::cout << "Iter: " << std::setw(6) << iter
                  << "  Time: " << std::fixed << std::setprecision(6) << t
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
          const auto u_prim = cons_to_prim(u_cons, eq.get_gamma());
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
