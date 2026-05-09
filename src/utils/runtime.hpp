#pragma once

#include <Kokkos_Core.hpp>

namespace DGSEM {

class KokkosSession {
public:
  KokkosSession() {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      owns_session_ = true;
    }
  }

  KokkosSession(int& argc, char** argv) {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize(argc, argv);
      owns_session_ = true;
    }
  }

  KokkosSession(const KokkosSession&) = delete;
  KokkosSession& operator=(const KokkosSession&) = delete;

  KokkosSession(KokkosSession&& other) noexcept
      : owns_session_(other.owns_session_) {
    other.owns_session_ = false;
  }

  KokkosSession& operator=(KokkosSession&& other) noexcept {
    if (this != &other) {
      finalize_if_owned();
      owns_session_ = other.owns_session_;
      other.owns_session_ = false;
    }
    return *this;
  }

  ~KokkosSession() { finalize_if_owned(); }

private:
  void finalize_if_owned() {
    if (owns_session_ && Kokkos::is_initialized()) {
      Kokkos::finalize();
    }
    owns_session_ = false;
  }

  bool owns_session_ = false;
};

template <class Basis>
class BasisGuard {
public:
  BasisGuard() { Basis::initialize(); }

  BasisGuard(const BasisGuard&) = delete;
  BasisGuard& operator=(const BasisGuard&) = delete;

  BasisGuard(BasisGuard&& other) noexcept : owns_basis_(other.owns_basis_) {
    other.owns_basis_ = false;
  }

  BasisGuard& operator=(BasisGuard&& other) noexcept {
    if (this != &other) {
      finalize_if_owned();
      owns_basis_ = other.owns_basis_;
      other.owns_basis_ = false;
    }
    return *this;
  }

  ~BasisGuard() { finalize_if_owned(); }

private:
  void finalize_if_owned() {
    if (owns_basis_) {
      Basis::finalize();
    }
    owns_basis_ = false;
  }

  bool owns_basis_ = true;
};

} // namespace DGSEM
