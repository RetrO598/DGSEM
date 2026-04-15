#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <dgsem.hpp>

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

    std::array<value_type, 2> domain_left = {-0.5, -0.5};
    std::array<value_type, 2> domain_right = {0.5, 0.5};

    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        domain_left, domain_right),
                    {false, true}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.3, 0.0001, false);

    Solution sol(mesh);
    AstroJetInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using Analyzer = DGSEM::CompositeAnalyzer<
        MyBasis, Eq, DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;
    using AnalyzerObserver =
        DGSEM::AnalyzerObserver<MyBasis, Eq, Solution, Analyzer>;
    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh, t_final);

    const value_type cfl = 0.1;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 800.0;
    // const value_type dt =
    //     cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);
    const value_type dt = 1e-8;

    time_integrator.add_observer(
        std::make_unique<AnalyzerObserver>(analyzer, sol, n_cells));
    time_integrator.add_observer(std::make_unique<PrintObserver>(1000));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "astro_jet_output", sol, container.node_coordinates, n_cells));
    time_integrator.solve(solver, sol, dt);

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
