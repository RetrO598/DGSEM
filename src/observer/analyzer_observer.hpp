#pragma once

#include <analyzer/analyzer.hpp>
#include <array>
#include <cstddef>
#include <iostream>
#include <observer/observer_base.hpp>
namespace DGSEM {

template <class Basis, class Equations, class Solution, class Analyzer>
class AnalyzerObserver : public SimulationObserver {
public:
  using trait = equations::EquationTraits<Equations>;
  constexpr static std::size_t NDIMS = trait::NDIMS;
  using T = typename trait::value_type;

  AnalyzerObserver(Analyzer& analyzer_, const Solution& sol_,
                   const std::array<std::size_t, NDIMS>& n_elems_)
      : analyzer(analyzer_), sol(sol_), n_elems(n_elems_) {}

  void on_step(int step, T time, bool& has_nan) override {
    AnalyzerFunctor<Basis, Equations, Analyzer, Solution>::apply(analyzer, sol,
                                                                 n_elems);

    if (analyzer.template get<DGSEM::DivergenceChecker<T, trait::NVARS>>()
            .has_nan) {
      std::cerr << "NaN detected at iter " << step << ", time " << time
                << std::endl;
      has_nan = true;
    }
  }

  void on_finish(int step, T time) override {}

private:
  Analyzer& analyzer;
  const Solution& sol;
  const std::array<std::size_t, NDIMS>& n_elems;
};
} // namespace DGSEM