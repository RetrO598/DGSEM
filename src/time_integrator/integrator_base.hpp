#pragma once

template <class T, class Solver, class Solution>
class TimeIntegrator {
public:
  virtual void step(Solver& solver, Solution& sol, T dt) = 0;

  T get_time() const { return time; }

protected:
  T time = static_cast<T>(0.0);
};
