#pragma once

namespace DGSEM {
class SimulationObserver {
public:
  virtual void on_step(int step, double time, bool& has_nan);
  virtual void on_finish(int step, double time);
  virtual ~SimulationObserver();
};
} // namespace DGSEM
