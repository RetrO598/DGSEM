#pragma once

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <observer/observer.hpp>

namespace DGSEM {

class PrintObserver : public SimulationObserver {
public:
  PrintObserver(int interval_ = 100) : interval(interval_) {}

  void on_step(int step, double time, bool& has_nan) override {
    if (step % interval == 0) {
      std::cout << "Iter: " << std::setw(5) << step
                << "  Time: " << std::scientific << std::setprecision(8) << time
                << std::endl;
    }
  }

  void on_finish(int step, double time) override {
    std::cout << "Finished at iter " << step << ", time " << time << std::endl;
  }

private:
  int interval = 100;
};
} // namespace DGSEM