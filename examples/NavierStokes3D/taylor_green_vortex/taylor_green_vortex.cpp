#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <dgsem.hpp>
#include <numbers>
#include <ostream>

template <class T>
struct TaylorGreenVortex3D
    : public DGSEM::AbstractInitial<
          TaylorGreenVortex3D<T>,
          DGSEM::equations::CompressibleNavierStokes3D<T>> {
  using Eq = DGSEM::equations::CompressibleNavierStokes3D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    const T x = coordinate[0];
    const T y = coordinate[1];
    const T z = coordinate[2];

    const T csp = std::sqrt(1.4 * 287.0 * 273.15);
    const T u0 = 0.1 * csp;

    const T rho = 1.0;
    const T v1 = u0 * std::sin(x) * std::cos(y) * std::cos(z);
    const T v2 = -u0 * std::cos(x) * std::sin(y) * std::cos(z);
    const T v3 = 0.0;
    const T p =
        static_cast<T>(287.0 * 273.15) +
        static_cast<T>(u0 * u0 / 16.0) *
            (std::cos(2.0 * x) * std::cos(2.0 * z) + 2.0 * std::cos(2.0 * y) +
             2.0 * std::cos(2.0 * x) + std::cos(2.0 * y) * std::cos(2.0 * z));

    return DGSEM::utils::prim_to_cons(std::array<T, 5>{rho, v1, v2, v3, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  Kokkos::initialize();
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleNavierStokes3D<value_type>;
    constexpr std::size_t order = 3;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, order>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux =
        DGSEM::VolumeIntegralSplitForm<MyBasis, Eq, DGSEM::ChandrashekarFlux>;
    using Mesh = DGSEM::StructuredMesh<value_type, 3>;

    const value_type Ma = 0.1;
    const value_type R = 287.0;
    const value_type gamma = 1.4;
    const value_type csp = std::sqrt(gamma * R * 273.15);
    const value_type u0 = Ma * csp;
    const value_type reynolds = 1600.0;
    const value_type mu = u0 / reynolds;
    const value_type prandtl = 0.72;

    MyBasis::initialize();

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;
    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    using DGSEM::PrintObserver;

    const std::size_t nx = 8;
    const std::size_t ny = 8;
    const std::size_t nz = 8;
    const value_type t_final = 1.0 / u0;

    std::cout << "t final: " << t_final << std::endl;

    const std::array<value_type, 3> domain_left = {0.0, 0.0, 0.0};
    const std::array<value_type, 3> domain_right = {
        2.0 * std::numbers::pi, 2.0 * std::numbers::pi, 2.0 * std::numbers::pi};
    const std::array<std::size_t, 3> n_cells = {nx, ny, nz};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq(gamma, mu, prandtl);

    DGSEM::StructuredElementContainer<value_type, 3> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 3>>, 3>
        initializer{DGSEM::LinearMapping<std::array<value_type, 3>>(
                        domain_left, domain_right),
                    {true, true, true}};
    initializer.init_elements(n_cells, container);
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    Solver solver(eq, mesh, container, boundaries);
    Solution sol(mesh);
    TaylorGreenVortex3D<value_type> initial{};
    solver.initialize(initial, sol);

    const value_type cfl = 0.1;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type dz = (domain_right[2] - domain_left[2]) / nz;
    const value_type max_speed = (u0 + csp);
    const value_type dt =
        cfl * std::min({dx, dy, dz}) / max_speed / (3.0 * (2 * order + 1));

    std::cout << "dt: " << dt << std::endl;

    TimeIntegrator time_integrator(sol, mesh, t_final);

    // using static_analyzer =
    //     DGSEM::AnalyzerWrapper<MyBasis, Eq, DGSEM::VolumeAverageEuler<Eq>>;

    // static_analyzer analyzer_statics;

    time_integrator.add_observer(std::make_unique<PrintObserver>(1000));
    // time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
    //     "navier_stokes_tgv_re1600", sol, container.node_coordinates, n_cells,
    //     1000));

    // time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
    //     DGSEM::VolumeWeightedAnalysisTag{}, analyzer_statics, sol, container,
    //     n_cells, DGSEM::VolumeAverageCsvWriter<Eq>("static.csv")));

    time_integrator.solve(solver, sol, dt);

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
