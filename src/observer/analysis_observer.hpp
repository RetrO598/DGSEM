#pragma once

#include <analyzer/analyzer.hpp>
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <observer/observer_base.hpp>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace DGSEM {

struct PointwiseAnalysisTag {};
struct VolumeWeightedAnalysisTag {};

inline constexpr PointwiseAnalysisTag pointwise_analysis{};
inline constexpr VolumeWeightedAnalysisTag volume_weighted_analysis{};

template <class Basis, class Equations, class Solution, class Analyzer>
struct PointwiseAnalysisRunner {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  const Solution& sol;
  std::array<std::size_t, traits::NDIMS> n_elems;

  void apply(Analyzer& analyzer) const {
    AnalyzerFunctor<Basis, Equations, Analyzer, Solution>::apply(analyzer, sol,
                                                                 n_elems);
  }
};

template <class Basis, class Equations, class Solution, class ElementContainer,
          class Analyzer>
struct VolumeWeightedAnalysisRunner {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  const Solution& sol;
  const ElementContainer& element_container;
  std::array<std::size_t, traits::NDIMS> n_elems;

  void apply(Analyzer& analyzer) const {
    VolumeWeightedAnalyzerFunctor<Basis, Equations, Analyzer, Solution,
                                  typename ElementContainer::ScalarArray>::
        apply(analyzer, sol, element_container.inverse_jacobian_device,
              n_elems);
  }
};

template <class Equations>
struct StopOnNaN {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;

  template <class Analyzer>
  void operator()(int step, T time, Analyzer& analyzer, bool& has_nan) const {
    if constexpr (requires(Analyzer a) {
                    a.template get<
                        DGSEM::DivergenceChecker<T, traits::NVARS>>();
                  }) {
      if (analyzer.template get<DGSEM::DivergenceChecker<T, traits::NVARS>>()
              .has_nan) {
        std::cerr << "NaN detected at iter " << step << ", time " << time
                  << std::endl;
        has_nan = true;
      }
    }
  }
};

template <class Equations>
class VolumeAverageCsvWriter {
public:
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using StatisticsAnalyzer = DGSEM::VolumeAverageEuler<Equations>;

  explicit VolumeAverageCsvWriter(std::string output_path_)
      : output_path(std::move(output_path_)), fout(output_path) {
    if (!fout) {
      std::cerr << "Failed to open file: " << output_path << std::endl;
      return;
    }

    fout << "step,time,volume_average_kinetic_energy,volume_average_entropy\n";
    fout << std::scientific << std::setprecision(16);
  }

  template <class Analyzer>
  void operator()(int step, T time, Analyzer& analyzer, bool&) {
    if (!fout) {
      return;
    }

    const auto& stats = analyzer.template get<StatisticsAnalyzer>();
    fout << step << ',' << time << ',' << stats.mean_kinetic_energy() << ','
         << stats.mean_entropy() << '\n';
    fout.flush();
  }

private:
  std::string output_path;
  std::ofstream fout;
};

template <class Runner, class Analyzer, class... Handlers>
class AnalysisObserver : public SimulationObserver {
public:
  using value_type = typename Runner::value_type;

  AnalysisObserver(Analyzer& analyzer_, Runner runner_, bool analyze_on_finish_,
                   Handlers... handlers_)
      : analyzer(analyzer_), runner(std::move(runner_)),
        analyze_on_finish(analyze_on_finish_),
        handlers(std::move(handlers_)...) {}

  void on_step(int step, double time, bool& has_nan) override {
    execute(step, static_cast<value_type>(time), has_nan);
  }

  void on_finish(int step, double time) override {
    if (!analyze_on_finish) {
      return;
    }

    bool has_nan = false;
    execute(step, static_cast<value_type>(time), has_nan);
  }

private:
  void execute(int step, value_type time, bool& has_nan) {
    runner.apply(analyzer);
    dispatch_handlers(step, time, has_nan,
                      std::index_sequence_for<Handlers...>{});
  }

  template <std::size_t... I>
  void dispatch_handlers(int step, value_type time, bool& has_nan,
                         std::index_sequence<I...>) {
    (std::get<I>(handlers)(step, time, analyzer, has_nan), ...);
  }

  Analyzer& analyzer;
  Runner runner;
  bool analyze_on_finish = false;
  std::tuple<Handlers...> handlers;
};

template <class Basis, class Equations, class Analyzer, class Solution,
          class... Handlers>
std::unique_ptr<SimulationObserver> make_analysis_observer(
    PointwiseAnalysisTag, Analyzer& analyzer, const Solution& sol,
    const std::array<std::size_t, equations::EquationTraits<Equations>::NDIMS>&
        n_elems,
    Handlers&&... handlers) {
  using Runner = PointwiseAnalysisRunner<Basis, Equations, Solution, Analyzer>;
  using Observer =
      AnalysisObserver<Runner, Analyzer, std::decay_t<Handlers>...>;

  return std::make_unique<Observer>(analyzer, Runner{sol, n_elems}, false,
                                    std::forward<Handlers>(handlers)...);
}

template <class Basis, class Equations, class Analyzer, class Solution,
          class ElementContainer, class... Handlers>
std::unique_ptr<SimulationObserver> make_analysis_observer(
    VolumeWeightedAnalysisTag, Analyzer& analyzer, const Solution& sol,
    const ElementContainer& element_container,
    const std::array<std::size_t, equations::EquationTraits<Equations>::NDIMS>&
        n_elems,
    Handlers&&... handlers) {
  using Runner = VolumeWeightedAnalysisRunner<Basis, Equations, Solution,
                                              ElementContainer, Analyzer>;
  using Observer =
      AnalysisObserver<Runner, Analyzer, std::decay_t<Handlers>...>;

  return std::make_unique<Observer>(analyzer,
                                    Runner{sol, element_container, n_elems},
                                    true, std::forward<Handlers>(handlers)...);
}

} // namespace DGSEM
