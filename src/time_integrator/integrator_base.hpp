#pragma once

template <class T, class Solver, class Solution>
class TimeIntegrator {
public:
  virtual void step(Solver &solver, Solution &sol, T dt) = 0;
  virtual void step_kokkos(Solver &solver, Solution &sol, T dt) = 0;
};