#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <dgsem.hpp>
#include <numbers>

template <class T>
struct SedovBlastWave
    : public DGSEM::AbstractInitial<SedovBlastWave<T>,
                                    DGSEM::equations::CompressibleEuler3D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler3D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x_norm = coordinate[0] - 0.0;
    T y_norm = coordinate[1] - 0.0;
    T z_norm = coordinate[2] - 0.0;

    T r = std::sqrt(x_norm * x_norm + y_norm * y_norm + z_norm * z_norm);

    T r0 = 0.21875;
    T E = 1.0;
    T p0_inner = 3.0 * (1.4 - 1.0) * E / (4.0 * std::numbers::pi * r0 * r0);
    T p0_outer = 1.0e-3;

    T rho = 1.0;
    T v1 = 0.0;
    T v2 = 0.0;
    T v3 = 0.0;

    T p = r > r0 ? p0_outer : p0_inner;

    return DGSEM::utils::prim_to_cons(std::array<T, 5>{rho, v1, v2, v3, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  DGSEM::KokkosSession kokkos;
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler3D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 3>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    // using VolumeFlux =
    //     DGSEM::VolumeIntegralSplitForm<MyBasis, Eq,
    //     DGSEM::ChandrashekarFlux>;
    using Mesh = DGSEM::StructuredMesh<value_type, 3>;

    DGSEM::BasisGuard<MyBasis> basis;

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 16;
    std::size_t ny = 16;
    std::size_t nz = 16;
    value_type t_final = 12.5;

    std::array<value_type, 3> domain_left = {0.0, 0.0, 0.0};
    std::array<value_type, 3> domain_right = {3.0, 3.0, 3.0};

    std::array<std::size_t, 3> n_cells = {nx, ny, nz};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 3> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 3>>, 3>
        initializer{DGSEM::LinearMapping<std::array<value_type, 3>>(
                        domain_left, domain_right),
                    {true, true, true}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(1.0, 0.001, false);

    Solution sol(mesh);
    SedovBlastWave<value_type> initial{};

    solver.initialize(initial, sol);

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    TimeIntegrator time_integrator(sol, mesh, t_final);

    const value_type cfl = 0.5;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 2.0;
    const value_type dt = 4e-3;
    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));

    time_integrator.add_observer(std::make_unique<PrintObserver>(100));

    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "blast_wave_hg_output", sol, container.node_coordinates, n_cells));
    time_integrator.solve(solver, sol, dt);

  }
  return 0;
}
