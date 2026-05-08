#include <observer/print_observer.hpp>

#include <iomanip>
#include <iostream>

namespace DGSEM {

PrintObserver::PrintObserver(int interval_) : interval(interval_) {}

void PrintObserver::on_step(int step, double time, bool&) {
  if (step % interval == 0) {
    std::cout << "Iter: " << std::setw(5) << step
              << "  Time: " << std::scientific << std::setprecision(8) << time
              << std::endl;
  }
}

void PrintObserver::on_finish(int step, double time) {
  std::cout << "Finished at iter " << step << ", time " << time << std::endl;
}

} // namespace DGSEM
