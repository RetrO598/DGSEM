#include <observer/observer_base.hpp>

namespace DGSEM {

void SimulationObserver::on_step(int, double, bool&) {}

void SimulationObserver::on_finish(int, double) {}

SimulationObserver::~SimulationObserver() = default;

} // namespace DGSEM
