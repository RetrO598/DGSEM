#pragma once

#include <observer/observer_base.hpp>

namespace DGSEM {

class PrintObserver : public SimulationObserver {
public:
  explicit PrintObserver(int interval_ = 100);

  void on_step(int step, double time, bool& has_nan) override;

  void on_finish(int step, double time) override;

private:
  int interval = 100;
};
} // namespace DGSEM
