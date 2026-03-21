#pragma once
#include "integrator_base.hpp"
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {

template <class T, class Solver, class Solution>
class SSPRK3 : public TimeIntegrator<T, Solver, Solution> {
public:
  explicit SSPRK3(const Solution &sol)
      : tmp1(sol.clone_shape()), tmp2(sol.clone_shape()) {};

  void step(Solver &solver, Solution &sol, T dt) override {
    solver.calc_rhs(sol);
    tmp1.u = sol.u + dt * sol.du;
    solver.calc_rhs(tmp1);
    tmp2.u = 3.0 / 4.0 * sol.u + 1.0 / 4.0 * tmp1.u + 1.0 / 4.0 * dt * tmp1.du;
    solver.calc_rhs(tmp2);
    sol.u = 1.0 / 3.0 * sol.u + 2.0 / 3.0 * tmp2.u + 2.0 / 3.0 * dt * tmp2.du;
  }

private:
  Solution tmp1;
  Solution tmp2;
};

} // namespace DGSEM