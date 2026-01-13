#pragma once
#include "equations.hpp"
#include <iostream>
namespace DGSEM {
enum class BoundaryCondition {
  Periodic,
  Wall,
  Inflow,
  Outflow,
  Extrapolate,
  Custom,
};

template <class Basis, class Equations, class BoundaryType>
struct BoundaryDispatcher;

template <class Basis, class Equations>
struct PeriodicBoundary {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  void operator()(const value_type &u) {
    std::cout << "Periodic Boundary Condition is not implemented yet." << "\n";
  }
};

template <class Basis, class Equations>
struct BoundaryDispatcher<Basis, Equations,
                          PeriodicBoundary<Equations, Basis>> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  static inline void apply(const value_type &u) {
    PeriodicBoundary<Equations, Basis> bc{};
    bc(u);
  }
};
} // namespace DGSEM
