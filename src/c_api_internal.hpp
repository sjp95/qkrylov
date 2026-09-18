#pragma once

#include "qkrylov/c_api.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <utility>

void set_last_error(const std::string& msg);
void set_last_error(const char* msg);

struct qkrylov_sector_t {
    bool use_sz = false;
    int sz2 = 0;
    bool use_nup = false;
    bool use_ndn = false;
    int nup = 0;
    int ndn = 0;
    bool use_n = false;
    int n = 0;
    bool use_nb = false;
    int nb = 0;
};

enum class BasisType { SpinHalf, SpinS, Fermion, Hubbard, TJ };

struct qkrylov_basis_t {
    BasisType type = BasisType::SpinHalf;
    int num_sites = 0;
    double spin_s = 0.5;
    qkrylov_sector_t sector;
    mutable std::shared_ptr<void> ptr32;
    mutable std::shared_ptr<void> ptr64;
    mutable uint64_t cached_dim = 0;
};

enum class SiteType { SpinHalf, SpinS, Fermion, Hubbard, TJ };

struct qkrylov_site_t {
    SiteType type = SiteType::SpinHalf;
    double spin_s = 0.5;
    mutable std::shared_ptr<void> ptr32;
    mutable std::shared_ptr<void> ptr64;
};

struct TermDescriptor {
    double coeff_real = 0.0;
    double coeff_imag = 0.0;
    std::vector<std::pair<std::string, int>> factors;
};

struct qkrylov_opsum_t {
    std::vector<TermDescriptor> terms;
};

struct qkrylov_hamiltonian_t {
    int precision = 1; // 0 = FP32, 1 = FP64
    uint64_t dim = 0;
    std::shared_ptr<void> impl;
};

struct qkrylov_device_vector_t {
    int precision = 1; // 0 = FP32, 1 = FP64
    uint64_t dim = 0;
    std::shared_ptr<void> impl;
};

struct qkrylov_ftlm_samples_t {
    int precision = 1; // 0 = FP32, 1 = FP64
    std::shared_ptr<void> impl;
};

