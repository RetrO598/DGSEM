#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <numbers>

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
    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p},
                                      static_cast<T>(1.4));
  }
};

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
    value_type t_final = 3.0;
    if (argc > 1)
      nx = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2)
      ny = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (argc > 3)
      t_final = std::strtod(argv[3], nullptr);

    std::array<value_type, 2> domain_left = {-1.0, -1.0};
    std::array<value_type, 2> domain_right = {1.0, 1.0};

    // std::array<value_type, 4> domain_mesh = {-1.0, 1.0, -1.0, 1.0};
    // std::array<std::array<value_type, 2>, 2> mapping_domain = {domain_left,
    //                                                            domain_right};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        domain_left, domain_right),
                    {true, true}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);

    Solution sol(mesh);
    KelvinHelmholtzInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh, t_final);

    const value_type cfl = 0.2;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 2.5;
    // const value_type dt =
    //     cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);
    const value_type dt = 8e-5;

    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));

    time_integrator.add_observer(std::make_unique<PrintObserver>(100));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "kelvin_helmholtz_hg_output", sol, container.node_coordinates,
        n_cells));
    time_integrator.solve(solver, sol, dt);

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
