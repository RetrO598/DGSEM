#pragma once

#include "observer/observer_base.hpp"
#include <memory>
#include <vector>

namespace DGSEM {
template <class T, class Solver, class Solution>
class TimeIntegrator {
public:
  TimeIntegrator() = default;

  TimeIntegrator(T final_time_) : final_time(final_time_) {}

  virtual void step(Solver& solver, Solution& sol, T dt) = 0;

  void add_observer(std::unique_ptr<SimulationObserver> observer) {
    observers.push_back(std::move(observer));
  }

  void solve(Solver& solver, Solution& sol, T dt) {
    while (time < final_time) {
      step(solver, sol, dt);
      time += dt;
      iter++;

      for (auto& obs : observers) {
        obs->on_step(iter, time, has_nan);
      }

      if (has_nan) {
        break;
      }
    }

    for (auto& obs : observers) {
      obs->on_finish(iter, time);
    }
  }

  T get_time() const { return time; }

protected:
  T time = static_cast<T>(0.0);
  T final_time = static_cast<T>(0.0);
  int iter = 0;
  bool has_nan = false;

private:
  std::vector<std::unique_ptr<SimulationObserver>> observers;
};
} // namespace DGSEM