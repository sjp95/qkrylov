#ifndef QKRYLOV_DOUBLE_PRECISION
#define QKRYLOV_DOUBLE_PRECISION
#endif

#include "qkrylov/c_api.h"
#include "c_api_internal.hpp"

#include "qkrylov/symmetry/sector.hpp"
#include "qkrylov/basis/basis.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/basis/fermion_basis.hpp"
#include "qkrylov/basis/hubbard_basis.hpp"
#include "qkrylov/basis/tj_basis.hpp"
#include "qkrylov/basis/spin_s_basis.hpp"
#include "qkrylov/sites/site.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/sites/spin_s_site.hpp"
#include "qkrylov/sites/fermion_site.hpp"
#include "qkrylov/sites/hubbard_site.hpp"
#include "qkrylov/sites/tj_site.hpp"
#include "qkrylov/core/device.hpp"

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <exception>

using namespace qkrylov;
using namespace qkrylov::fp64;

namespace {

inline std::shared_ptr<Basis> make_basis_fp64(const qkrylov_basis_t& b) {
    Sector sec;
    sec.use_sz = b.sector.use_sz;
    sec.sz2 = b.sector.sz2;
    sec.use_nup = b.sector.use_nup;
    sec.use_ndn = b.sector.use_ndn;
    sec.nup = b.sector.nup;
    sec.ndn = b.sector.ndn;
    sec.use_n = b.sector.use_n;
    sec.n = b.sector.n;
    sec.use_nb = b.sector.use_nb;
    sec.nb = b.sector.nb;

    switch (b.type) {
        case BasisType::SpinHalf:
            return std::make_shared<SpinHalfBasis>(b.num_sites, sec);
        case BasisType::SpinS:
            return std::make_shared<SpinSBasis>(b.num_sites, b.spin_s, sec);
        case BasisType::Fermion:
            return std::make_shared<FermionBasis>(b.num_sites, sec);
        case BasisType::Hubbard:
            return std::make_shared<HubbardBasis>(b.num_sites, sec);
        case BasisType::TJ:
            return std::make_shared<TJBasis>(b.num_sites, sec);
    }
    return nullptr;
}

inline std::shared_ptr<Basis> get_or_create_basis_fp64(const qkrylov_basis_t& b) {
    if (b.ptr64) {
        return std::static_pointer_cast<Basis>(b.ptr64);
    }
    auto ptr = make_basis_fp64(b);
    b.ptr64 = ptr;
    if (ptr) b.cached_dim = ptr->size();
    return ptr;
}

inline std::shared_ptr<Site> make_site_fp64(const qkrylov_site_t& s) {
    switch (s.type) {
        case SiteType::SpinHalf:
            return std::make_shared<SpinHalfSite>();
        case SiteType::SpinS:
            return std::make_shared<SpinSSite>(s.spin_s);
        case SiteType::Fermion:
            return std::make_shared<FermionSite>();
        case SiteType::Hubbard:
            return std::make_shared<HubbardSite>();
        case SiteType::TJ:
            return std::make_shared<TJSite>();
    }
    return nullptr;
}

inline std::shared_ptr<Site> get_or_create_site_fp64(const qkrylov_site_t& s) {
    if (s.ptr64) {
        return std::static_pointer_cast<Site>(s.ptr64);
    }
    auto ptr = make_site_fp64(s);
    s.ptr64 = ptr;
    return ptr;
}

} // anonymous namespace

thread_local std::string g_last_error_message;

void set_last_error(const std::string& msg) {
    g_last_error_message = msg;
}

void set_last_error(const char* msg) {
    g_last_error_message = msg ? msg : "";
}

extern "C" {

/* -----------------------------------------------------------------------------
 * Error Diagnostics API
 * ----------------------------------------------------------------------------- */
QKRYLOV_API const char* qkrylov_get_last_error_message(void) {
    return g_last_error_message.c_str();
}

QKRYLOV_API void qkrylov_clear_last_error(void) {
    g_last_error_message.clear();
}

/* -----------------------------------------------------------------------------
 * Sector API
 * ----------------------------------------------------------------------------- */
qkrylov_sector_h qkrylov_sector_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_sector_t>();
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_sector_create");
        return nullptr;
    }
}

void qkrylov_sector_destroy(qkrylov_sector_h sector) {
    if (sector) delete sector;
}

int qkrylov_sector_set_sz(qkrylov_sector_h sector, int sz2) {
    if (!sector) {
        set_last_error("qkrylov_sector_set_sz: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        sector->use_sz = true;
        sector->sz2 = sz2;
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_sector_set_sz");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_hubbard_particles(qkrylov_sector_h sector, int nup, int ndn) {
    if (!sector) {
        set_last_error("qkrylov_sector_set_hubbard_particles: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        sector->use_nup = true;
        sector->use_ndn = true;
        sector->nup = nup;
        sector->ndn = ndn;
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_sector_set_hubbard_particles");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_n(qkrylov_sector_h sector, int n) {
    if (!sector) {
        set_last_error("qkrylov_sector_set_n: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        sector->use_n = true;
        sector->n = n;
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_sector_set_n");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_set_nb(qkrylov_sector_h sector, int nb) {
    if (!sector) {
        set_last_error("qkrylov_sector_set_nb: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        sector->use_nb = true;
        sector->nb = nb;
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_sector_set_nb");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_sector_get_sz(qkrylov_sector_h sector, int* sz2_out, int* active_out) {
    if (!sector) {
        set_last_error("qkrylov_sector_get_sz: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (active_out) *active_out = sector->use_sz ? 1 : 0;
    if (sz2_out) *sz2_out = sector->sz2;
    return QKRYLOV_SUCCESS;
}

int qkrylov_sector_get_hubbard_particles(qkrylov_sector_h sector, int* nup_out, int* ndn_out, int* active_out) {
    if (!sector) {
        set_last_error("qkrylov_sector_get_hubbard_particles: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (active_out) *active_out = (sector->use_nup && sector->use_ndn) ? 1 : 0;
    if (nup_out) *nup_out = sector->nup;
    if (ndn_out) *ndn_out = sector->ndn;
    return QKRYLOV_SUCCESS;
}

int qkrylov_sector_get_n(qkrylov_sector_h sector, int* n_out, int* active_out) {
    if (!sector) {
        set_last_error("qkrylov_sector_get_n: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (active_out) *active_out = sector->use_n ? 1 : 0;
    if (n_out) *n_out = sector->n;
    return QKRYLOV_SUCCESS;
}

int qkrylov_sector_get_nb(qkrylov_sector_h sector, int* nb_out, int* active_out) {
    if (!sector) {
        set_last_error("qkrylov_sector_get_nb: sector handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (active_out) *active_out = sector->use_nb ? 1 : 0;
    if (nb_out) *nb_out = sector->nb;
    return QKRYLOV_SUCCESS;
}

/* -----------------------------------------------------------------------------
 * Basis API
 * ----------------------------------------------------------------------------- */
qkrylov_basis_h qkrylov_spinhalf_basis_create(int num_sites, qkrylov_sector_h sector) {
    if (num_sites <= 0) {
        set_last_error("qkrylov_spinhalf_basis_create: num_sites must be positive");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        handle->type = BasisType::SpinHalf;
        handle->num_sites = num_sites;
        handle->spin_s = 0.5;
        if (sector) handle->sector = *sector;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_spinhalf_basis_create");
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_basis_create_spin_s(int N, double S, const qkrylov_sector_t* sector) {
    if (N <= 0 || S <= 0.0) {
        set_last_error("qkrylov_basis_create_spin_s: num_sites and spin S must be positive");
        return nullptr;
    }
    int two_s = static_cast<int>(std::round(2.0 * S));
    if (std::abs(2.0 * S - two_s) > 1e-6) {
        set_last_error("qkrylov_basis_create_spin_s: spin S must be an integer or half-integer");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        handle->type = BasisType::SpinS;
        handle->num_sites = N;
        handle->spin_s = S;
        if (sector) handle->sector = *sector;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_create_spin_s");
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_fermion_basis_create(int num_sites, qkrylov_sector_h sector) {
    if (num_sites <= 0) {
        set_last_error("qkrylov_fermion_basis_create: num_sites must be positive");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        handle->type = BasisType::Fermion;
        handle->num_sites = num_sites;
        handle->spin_s = 0.5;
        if (sector) handle->sector = *sector;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_fermion_basis_create");
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_hubbard_basis_create(int num_sites, qkrylov_sector_h sector) {
    if (num_sites <= 0) {
        set_last_error("qkrylov_hubbard_basis_create: num_sites must be positive");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        handle->type = BasisType::Hubbard;
        handle->num_sites = num_sites;
        handle->spin_s = 0.5;
        if (sector) handle->sector = *sector;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hubbard_basis_create");
        return nullptr;
    }
}

qkrylov_basis_h qkrylov_tj_basis_create(int num_sites, qkrylov_sector_h sector) {
    if (num_sites <= 0) {
        set_last_error("qkrylov_tj_basis_create: num_sites must be positive");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_basis_t>();
        handle->type = BasisType::TJ;
        handle->num_sites = num_sites;
        handle->spin_s = 0.5;
        if (sector) handle->sector = *sector;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_tj_basis_create");
        return nullptr;
    }
}

void qkrylov_basis_destroy(qkrylov_basis_h basis) {
    if (basis) delete basis;
}

uint64_t qkrylov_basis_dimension(qkrylov_basis_h basis) {
    if (!basis) {
        set_last_error("qkrylov_basis_dimension: basis handle is null");
        return 0;
    }
    if (basis->cached_dim > 0) return basis->cached_dim;
    try {
        auto b_ptr = get_or_create_basis_fp64(*basis);
        return b_ptr ? b_ptr->size() : 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return 0;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_dimension");
        return 0;
    }
}

int qkrylov_basis_nsites(qkrylov_basis_h basis) {
    if (!basis) {
        set_last_error("qkrylov_basis_nsites: basis handle is null");
        return 0;
    }
    return basis->num_sites;
}

uint64_t qkrylov_basis_state(qkrylov_basis_h basis, uint64_t index) {
    if (!basis) {
        set_last_error("qkrylov_basis_state: basis handle is null");
        return 0;
    }
    try {
        auto b_ptr = get_or_create_basis_fp64(*basis);
        return b_ptr ? b_ptr->state(index) : 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return 0;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_state");
        return 0;
    }
}

int64_t qkrylov_basis_index(qkrylov_basis_h basis, uint64_t state_bitstring) {
    if (!basis) {
        set_last_error("qkrylov_basis_index: basis handle is null");
        return -1;
    }
    try {
        auto b_ptr = get_or_create_basis_fp64(*basis);
        return b_ptr ? static_cast<int64_t>(b_ptr->index(state_bitstring)) : -1;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return -1;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_index");
        return -1;
    }
}

int qkrylov_basis_contains(qkrylov_basis_h basis, uint64_t state_bitstring) {
    if (!basis) {
        set_last_error("qkrylov_basis_contains: basis handle is null");
        return 0;
    }
    try {
        auto b_ptr = get_or_create_basis_fp64(*basis);
        return (b_ptr && b_ptr->contains(state_bitstring)) ? 1 : 0;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return 0;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_contains");
        return 0;
    }
}

int qkrylov_basis_get_type(qkrylov_basis_h basis, qkrylov_basis_type_t* type_out) {
    if (!basis) {
        set_last_error("qkrylov_basis_get_type: basis handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!type_out) {
        set_last_error("qkrylov_basis_get_type: type_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    *type_out = static_cast<qkrylov_basis_type_t>(basis->type);
    return QKRYLOV_SUCCESS;
}

int qkrylov_basis_get_spin(qkrylov_basis_h basis, double* spin_out) {
    if (!basis) {
        set_last_error("qkrylov_basis_get_spin: basis handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!spin_out) {
        set_last_error("qkrylov_basis_get_spin: spin_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    switch (basis->type) {
        case BasisType::SpinHalf:
            *spin_out = 0.5;
            break;
        case BasisType::SpinS:
            *spin_out = basis->spin_s;
            break;
        case BasisType::Fermion:
            *spin_out = 0.0;
            break;
        case BasisType::Hubbard:
        case BasisType::TJ:
            *spin_out = 0.5;
            break;
    }
    return QKRYLOV_SUCCESS;
}

int qkrylov_basis_get_dimension_per_site(qkrylov_basis_h basis, int* d_out) {
    if (!basis) {
        set_last_error("qkrylov_basis_get_dimension_per_site: basis handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!d_out) {
        set_last_error("qkrylov_basis_get_dimension_per_site: d_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    switch (basis->type) {
        case BasisType::SpinHalf:
            *d_out = 2;
            break;
        case BasisType::SpinS:
            *d_out = static_cast<int>(std::round(2.0 * basis->spin_s + 1.0));
            break;
        case BasisType::Fermion:
            *d_out = 2;
            break;
        case BasisType::Hubbard:
            *d_out = 4;
            break;
        case BasisType::TJ:
            *d_out = 3;
            break;
    }
    return QKRYLOV_SUCCESS;
}

qkrylov_sector_h qkrylov_basis_get_sector(qkrylov_basis_h basis) {
    if (!basis) {
        set_last_error("qkrylov_basis_get_sector: basis handle is null");
        return nullptr;
    }
    try {
        auto sec = std::make_unique<qkrylov_sector_t>(basis->sector);
        return sec.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_basis_get_sector");
        return nullptr;
    }
}

/* -----------------------------------------------------------------------------
 * Site API
 * ----------------------------------------------------------------------------- */
qkrylov_site_h qkrylov_spinhalf_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->type = SiteType::SpinHalf;
        handle->spin_s = 0.5;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_spinhalf_site_create");
        return nullptr;
    }
}

qkrylov_site_h qkrylov_site_create_spin_s(double S) {
    if (S <= 0.0) {
        set_last_error("qkrylov_site_create_spin_s: spin S must be positive");
        return nullptr;
    }
    int two_s = static_cast<int>(std::round(2.0 * S));
    if (std::abs(2.0 * S - two_s) > 1e-6) {
        set_last_error("qkrylov_site_create_spin_s: spin S must be an integer or half-integer");
        return nullptr;
    }
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->type = SiteType::SpinS;
        handle->spin_s = S;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_site_create_spin_s");
        return nullptr;
    }
}

qkrylov_site_h qkrylov_fermion_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->type = SiteType::Fermion;
        handle->spin_s = 0.5;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_fermion_site_create");
        return nullptr;
    }
}

qkrylov_site_h qkrylov_hubbard_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->type = SiteType::Hubbard;
        handle->spin_s = 0.5;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_hubbard_site_create");
        return nullptr;
    }
}

qkrylov_site_h qkrylov_tj_site_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_site_t>();
        handle->type = SiteType::TJ;
        handle->spin_s = 0.5;
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_tj_site_create");
        return nullptr;
    }
}

void qkrylov_site_destroy(qkrylov_site_h site) {
    if (site) delete site;
}

int qkrylov_site_apply(
    qkrylov_site_h site,
    const char* op,
    int site_idx,
    uint64_t state,
    qkrylov_local_action_t* action_out
) {
    if (!site) {
        set_last_error("qkrylov_site_apply: site handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!op) {
        set_last_error("qkrylov_site_apply: operator name is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!action_out) {
        set_last_error("qkrylov_site_apply: action_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (site_idx < 0) {
        set_last_error("qkrylov_site_apply: site index must be non-negative");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        auto s_ptr = get_or_create_site_fp64(*site);
        if (!s_ptr) {
            set_last_error("qkrylov_site_apply: failed to create site instance");
            return QKRYLOV_ERROR_EXCEPTION;
        }
        auto act = s_ptr->apply(std::string(op), site_idx, state);
        action_out->valid = act.valid ? 1 : 0;
        action_out->new_state = act.new_state;
        action_out->matrix_element_re = act.matrix_element.real();
        action_out->matrix_element_im = act.matrix_element.imag();
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_site_apply");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_site_get_type(qkrylov_site_h site, qkrylov_site_type_t* type_out) {
    if (!site) {
        set_last_error("qkrylov_site_get_type: site handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!type_out) {
        set_last_error("qkrylov_site_get_type: type_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    *type_out = static_cast<qkrylov_site_type_t>(site->type);
    return QKRYLOV_SUCCESS;
}

int qkrylov_site_get_spin(qkrylov_site_h site, double* spin_out) {
    if (!site) {
        set_last_error("qkrylov_site_get_spin: site handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!spin_out) {
        set_last_error("qkrylov_site_get_spin: spin_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    switch (site->type) {
        case SiteType::SpinHalf:
            *spin_out = 0.5;
            break;
        case SiteType::SpinS:
            *spin_out = site->spin_s;
            break;
        case SiteType::Fermion:
            *spin_out = 0.0;
            break;
        case SiteType::Hubbard:
        case SiteType::TJ:
            *spin_out = 0.5;
            break;
    }
    return QKRYLOV_SUCCESS;
}

int qkrylov_site_get_dimension_per_site(qkrylov_site_h site, int* d_out) {
    if (!site) {
        set_last_error("qkrylov_site_get_dimension_per_site: site handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!d_out) {
        set_last_error("qkrylov_site_get_dimension_per_site: d_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    switch (site->type) {
        case SiteType::SpinHalf:
            *d_out = 2;
            break;
        case SiteType::SpinS:
            *d_out = static_cast<int>(std::round(2.0 * site->spin_s + 1.0));
            break;
        case SiteType::Fermion:
            *d_out = 2;
            break;
        case SiteType::Hubbard:
            *d_out = 4;
            break;
        case SiteType::TJ:
            *d_out = 3;
            break;
    }
    return QKRYLOV_SUCCESS;
}

/* -----------------------------------------------------------------------------
 * OpSum API
 * ----------------------------------------------------------------------------- */
qkrylov_opsum_h qkrylov_opsum_create(void) {
    try {
        auto handle = std::make_unique<qkrylov_opsum_t>();
        return handle.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_create");
        return nullptr;
    }
}

void qkrylov_opsum_destroy(qkrylov_opsum_h opsum) {
    if (opsum) delete opsum;
}

int qkrylov_opsum_clear(qkrylov_opsum_h opsum) {
    if (!opsum) {
        set_last_error("qkrylov_opsum_clear: opsum handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    opsum->terms.clear();
    return QKRYLOV_SUCCESS;
}

int qkrylov_opsum_size(qkrylov_opsum_h opsum, int* size_out) {
    if (!opsum) {
        set_last_error("qkrylov_opsum_size: opsum handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (!size_out) {
        set_last_error("qkrylov_opsum_size: size_out pointer is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    *size_out = static_cast<int>(opsum->terms.size());
    return QKRYLOV_SUCCESS;
}

int qkrylov_opsum_get_term_info(
    qkrylov_opsum_h opsum,
    int term_idx,
    double* coeff_re_out,
    double* coeff_im_out,
    int* num_factors_out
) {
    if (!opsum) {
        set_last_error("qkrylov_opsum_get_term_info: opsum handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (term_idx < 0 || term_idx >= static_cast<int>(opsum->terms.size())) {
        set_last_error("qkrylov_opsum_get_term_info: term_idx out of range");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    const auto& t = opsum->terms[term_idx];
    if (coeff_re_out) *coeff_re_out = t.coeff_real;
    if (coeff_im_out) *coeff_im_out = t.coeff_imag;
    if (num_factors_out) *num_factors_out = static_cast<int>(t.factors.size());
    return QKRYLOV_SUCCESS;
}

int qkrylov_opsum_get_factor(
    qkrylov_opsum_h opsum,
    int term_idx,
    int factor_idx,
    char* op_buf,
    int op_buf_len,
    int* site_out
) {
    if (!opsum) {
        set_last_error("qkrylov_opsum_get_factor: opsum handle is null");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    if (term_idx < 0 || term_idx >= static_cast<int>(opsum->terms.size())) {
        set_last_error("qkrylov_opsum_get_factor: term_idx out of range");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    const auto& t = opsum->terms[term_idx];
    if (factor_idx < 0 || factor_idx >= static_cast<int>(t.factors.size())) {
        set_last_error("qkrylov_opsum_get_factor: factor_idx out of range");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    const auto& f = t.factors[factor_idx];
    if (op_buf && op_buf_len > 0) {
        std::snprintf(op_buf, op_buf_len, "%s", f.first.c_str());
    }
    if (site_out) {
        *site_out = f.second;
    }
    return QKRYLOV_SUCCESS;
}

int qkrylov_opsum_add_term_1body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                     const char* op1, int site1) {
    if (!opsum || !op1) {
        set_last_error("qkrylov_opsum_add_term_1body_fp32: null opsum or operator name");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        opsum->terms.push_back({static_cast<double>(coeff_real), static_cast<double>(coeff_imag),
                                {{std::string(op1), site1}}});
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_1body_fp32");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_1body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                     const char* op1, int site1) {
    if (!opsum || !op1) {
        set_last_error("qkrylov_opsum_add_term_1body_fp64: null opsum or operator name");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        opsum->terms.push_back({coeff_real, coeff_imag, {{std::string(op1), site1}}});
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_1body_fp64");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_1body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                const char* op1, int site1) {
    return qkrylov_opsum_add_term_1body_fp64(opsum, coeff_real, coeff_imag, op1, site1);
}

int qkrylov_opsum_add_term_2body_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                     const char* op1, int site1,
                                     const char* op2, int site2) {
    if (!opsum || !op1 || !op2) {
        set_last_error("qkrylov_opsum_add_term_2body_fp32: null opsum or operator name");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        opsum->terms.push_back({static_cast<double>(coeff_real), static_cast<double>(coeff_imag),
                                {{std::string(op1), site1}, {std::string(op2), site2}}});
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_2body_fp32");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_2body_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                     const char* op1, int site1,
                                     const char* op2, int site2) {
    if (!opsum || !op1 || !op2) {
        set_last_error("qkrylov_opsum_add_term_2body_fp64: null opsum or operator name");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        opsum->terms.push_back({coeff_real, coeff_imag,
                                {{std::string(op1), site1}, {std::string(op2), site2}}});
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_2body_fp64");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_2body(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                const char* op1, int site1,
                                const char* op2, int site2) {
    return qkrylov_opsum_add_term_2body_fp64(opsum, coeff_real, coeff_imag, op1, site1, op2, site2);
}

int qkrylov_opsum_add_term_nbody_fp32(qkrylov_opsum_h opsum, float coeff_real, float coeff_imag,
                                     int n_factors, const char** ops, const int* sites) {
    if (!opsum || !ops || !sites || n_factors <= 0) {
        set_last_error("qkrylov_opsum_add_term_nbody_fp32: null pointer or invalid n_factors");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        TermDescriptor term;
        term.coeff_real = static_cast<double>(coeff_real);
        term.coeff_imag = static_cast<double>(coeff_imag);
        for (int i = 0; i < n_factors; ++i) {
            if (!ops[i]) {
                set_last_error("qkrylov_opsum_add_term_nbody_fp32: operator name at factor is null");
                return QKRYLOV_ERROR_INVALID_ARG;
            }
            term.factors.push_back({std::string(ops[i]), sites[i]});
        }
        opsum->terms.push_back(std::move(term));
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_nbody_fp32");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_nbody_fp64(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                     int n_factors, const char** ops, const int* sites) {
    if (!opsum || !ops || !sites || n_factors <= 0) {
        set_last_error("qkrylov_opsum_add_term_nbody_fp64: null pointer or invalid n_factors");
        return QKRYLOV_ERROR_INVALID_ARG;
    }
    try {
        TermDescriptor term;
        term.coeff_real = coeff_real;
        term.coeff_imag = coeff_imag;
        for (int i = 0; i < n_factors; ++i) {
            if (!ops[i]) {
                set_last_error("qkrylov_opsum_add_term_nbody_fp64: operator name at factor is null");
                return QKRYLOV_ERROR_INVALID_ARG;
            }
            term.factors.push_back({std::string(ops[i]), sites[i]});
        }
        opsum->terms.push_back(std::move(term));
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_opsum_add_term_nbody_fp64");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

int qkrylov_opsum_add_term_nbody(qkrylov_opsum_h opsum, double coeff_real, double coeff_imag,
                                int n_factors, const char** ops, const int* sites) {
    return qkrylov_opsum_add_term_nbody_fp64(opsum, coeff_real, coeff_imag, n_factors, ops, sites);
}

/* -----------------------------------------------------------------------------
 * Device & Hardware Query API
 * ----------------------------------------------------------------------------- */
int qkrylov_is_gpu_build(void) {
    return Device::is_gpu_build() ? 1 : 0;
}

const char* qkrylov_find_gpu(void) {
    if (Device::is_gpu_build()) {
        static std::string backend = Device::backend_name();
        return backend.c_str();
    }
    return nullptr;
}

int qkrylov_gpu_count(void) {
    return Device::gpu_count();
}

int qkrylov_initialize_device(const char* device_str) {
    try {
        detail::initialize_kokkos(Device(device_str ? device_str : "cpu"));
        return QKRYLOV_SUCCESS;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return QKRYLOV_ERROR_EXCEPTION;
    } catch (...) {
        set_last_error("Unknown exception in qkrylov_initialize_device");
        return QKRYLOV_ERROR_EXCEPTION;
    }
}

/* -----------------------------------------------------------------------------
 * Common MatrixFreeHamiltonian API
 * ----------------------------------------------------------------------------- */
void qkrylov_hamiltonian_destroy(qkrylov_hamiltonian_h h) {
    if (h) delete h;
}

uint64_t qkrylov_hamiltonian_dimension(qkrylov_hamiltonian_h h) {
    if (!h) {
        set_last_error("qkrylov_hamiltonian_dimension: hamiltonian handle is null");
        return 0;
    }
    return h->dim;
}

int qkrylov_hamiltonian_precision(qkrylov_hamiltonian_h h) {
    if (!h) {
        set_last_error("qkrylov_hamiltonian_precision: hamiltonian handle is null");
        return -1;
    }
    return h->precision;
}

/* Default (FP64) Convenience Aliases */
qkrylov_hamiltonian_h qkrylov_hamiltonian_create(qkrylov_basis_h basis,
                                                qkrylov_site_h site,
                                                qkrylov_opsum_h opsum) {
    return qkrylov_hamiltonian_create_fp64(basis, site, opsum);
}

qkrylov_hamiltonian_h qkrylov_hamiltonian_create_device(qkrylov_basis_h basis,
                                                        qkrylov_site_h site,
                                                        qkrylov_opsum_h opsum,
                                                        const char* device_str) {
    return qkrylov_hamiltonian_create_device_fp64(basis, site, opsum, device_str);
}

int qkrylov_hamiltonian_apply(qkrylov_hamiltonian_h h,
                              const double* x_real, const double* x_imag,
                              double* y_real, double* y_imag) {
    return qkrylov_hamiltonian_apply_fp64(h, x_real, x_imag, y_real, y_imag);
}

int qkrylov_hamiltonian_apply_complex(qkrylov_hamiltonian_h h,
                                      const double* x_complex,
                                      double* y_complex) {
    return qkrylov_hamiltonian_apply_complex_fp64(h, x_complex, y_complex);
}

int qkrylov_hamiltonian_apply_device(qkrylov_hamiltonian_h h,
                                     const qkrylov_device_vector_h x_dev,
                                     qkrylov_device_vector_h y_dev) {
    return qkrylov_hamiltonian_apply_device_fp64(h, x_dev, y_dev);
}

int qkrylov_hamiltonian_diagonal(qkrylov_hamiltonian_h h, double* diag_out) {
    return qkrylov_hamiltonian_diagonal_fp64(h, diag_out);
}

int qkrylov_hamiltonian_diagonal_device(qkrylov_hamiltonian_h h,
                                        qkrylov_device_vector_h diag_out) {
    return qkrylov_hamiltonian_diagonal_device_fp64(h, diag_out);
}

int qkrylov_lanczos_ground_state(qkrylov_hamiltonian_h h,
                                 int maxiter,
                                 double tol,
                                 qkrylov_lanczos_result_c_t* result) {
    return qkrylov_lanczos_ground_state_fp64(h, maxiter, tol, result);
}

int qkrylov_lanczos_ground_state_complex(qkrylov_hamiltonian_h h,
                                         int maxiter,
                                         double tol,
                                         qkrylov_lanczos_result_c_t* result,
                                         double* eigenvector_complex) {
    return qkrylov_lanczos_ground_state_complex_fp64(h, maxiter, tol, result, eigenvector_complex);
}

int qkrylov_lanczos_two_pass_ground_state(qkrylov_hamiltonian_h h,
                                          int maxiter,
                                          double tol,
                                          qkrylov_lanczos_result_c_t* result) {
    return qkrylov_lanczos_two_pass_ground_state_fp64(h, maxiter, tol, result);
}

int qkrylov_lanczos_two_pass_ground_state_complex(qkrylov_hamiltonian_h h,
                                                  int maxiter,
                                                  double tol,
                                                  qkrylov_lanczos_result_c_t* result,
                                                  double* eigenvector_complex) {
    return qkrylov_lanczos_two_pass_ground_state_complex_fp64(h, maxiter, tol, result, eigenvector_complex);
}

int qkrylov_lanczos_lowest_complex(qkrylov_hamiltonian_h h,
                                   int n_eig,
                                   int maxiter,
                                   double tol,
                                   double* eigenvalues_out,
                                   double* eigenvectors_complex_out,
                                   qkrylov_lanczos_lowest_result_c_t* result_info,
                                   const double* initial_vector_complex) {
    return qkrylov_lanczos_lowest_complex_fp64(h, n_eig, maxiter, tol, eigenvalues_out, eigenvectors_complex_out, result_info, initial_vector_complex);
}

int qkrylov_davidson_lowest_complex(qkrylov_hamiltonian_h h,
                                    int n_eig,
                                    int max_subspace,
                                    double tol,
                                    double* eigenvalues_out,
                                    double* eigenvectors_complex_out,
                                    qkrylov_davidson_result_c_t* result_info) {
    return qkrylov_davidson_lowest_complex_fp64(h, n_eig, max_subspace, tol, eigenvalues_out, eigenvectors_complex_out, result_info);
}

int qkrylov_continued_fraction_coeffs_complex(qkrylov_hamiltonian_h h,
                                              const double* phi0_complex,
                                              int n_iter,
                                              double* alphas_out,
                                              double* betas_out,
                                              double* norm_phi0_out,
                                              int* num_coeffs_out) {
    return qkrylov_continued_fraction_coeffs_complex_fp64(h, phi0_complex, n_iter, alphas_out, betas_out, norm_phi0_out, num_coeffs_out);
}

double qkrylov_evaluate_spectral_function(const double* alphas,
                                          const double* betas,
                                          size_t n,
                                          double norm_phi0,
                                          double omega,
                                          double E0,
                                          double eta) {
    return qkrylov_evaluate_spectral_function_fp64(alphas, betas, n, norm_phi0, omega, E0, eta);
}

int qkrylov_ftlm(qkrylov_hamiltonian_h h,
                 double beta,
                 int n_random,
                 int n_steps,
                 qkrylov_ftlm_result_c_t* result) {
    return qkrylov_ftlm_fp64(h, beta, n_random, n_steps, result);
}

int qkrylov_ftlm_sweep(qkrylov_hamiltonian_h h,
                       const double* beta_grid,
                       int num_betas,
                       const qkrylov_hamiltonian_h* observables,
                       int num_observables,
                       int n_random,
                       int n_steps,
                       uint64_t seed,
                       qkrylov_ftlm_sweep_result_c_t* result) {
    return qkrylov_ftlm_sweep_fp64(h, beta_grid, num_betas, observables, num_observables, n_random, n_steps, seed, result);
}

void qkrylov_ftlm_sweep_result_free(qkrylov_ftlm_sweep_result_c_t* result) {
    qkrylov_ftlm_sweep_result_free_fp64(result);
}

int qkrylov_solver_correction_vector(qkrylov_hamiltonian_h h,
                                     const double* op_psi0_complex,
                                     double e0,
                                     double omega,
                                     double eta,
                                     int max_iter,
                                     double tol,
                                     qkrylov_correction_vector_result_c_t* result,
                                     double* correction_vector_out_complex) {
    return qkrylov_solver_correction_vector_fp64(h, op_psi0_complex, e0, omega, eta, max_iter, tol, result, correction_vector_out_complex);
}

int qkrylov_vector_dot(uint64_t dim, const double* x_complex, const double* y_complex, double* dot_re, double* dot_im) {
    return qkrylov_vector_dot_fp64(dim, x_complex, y_complex, dot_re, dot_im);
}

int qkrylov_vector_norm(uint64_t dim, const double* x_complex, double* norm_out) {
    return qkrylov_vector_norm_fp64(dim, x_complex, norm_out);
}

int qkrylov_vector_axpy(uint64_t dim, double a_re, double a_im, const double* x_complex, double* y_complex) {
    return qkrylov_vector_axpy_fp64(dim, a_re, a_im, x_complex, y_complex);
}

int qkrylov_vector_scal(uint64_t dim, double a_re, double a_im, double* x_complex) {
    return qkrylov_vector_scal_fp64(dim, a_re, a_im, x_complex);
}

int qkrylov_vector_normalize(uint64_t dim, double* x_complex) {
    return qkrylov_vector_normalize_fp64(dim, x_complex);
}

int qkrylov_vector_zero_fill(uint64_t dim, double* x_complex) {
    return qkrylov_vector_zero_fill_fp64(dim, x_complex);
}

int qkrylov_vector_copy(uint64_t dim, const double* src_complex, double* dst_complex) {
    return qkrylov_vector_copy_fp64(dim, src_complex, dst_complex);
}

/* -----------------------------------------------------------------------------
 * Device-Resident Vector Management & Aliases
 * ----------------------------------------------------------------------------- */
void qkrylov_device_vector_destroy(qkrylov_device_vector_h vec) {
    if (vec) delete vec;
}

uint64_t qkrylov_device_vector_dimension(qkrylov_device_vector_h vec) {
    if (!vec) {
        set_last_error("qkrylov_device_vector_dimension: vector handle is null");
        return 0;
    }
    return vec->dim;
}

int qkrylov_device_vector_precision(qkrylov_device_vector_h vec) {
    if (!vec) {
        set_last_error("qkrylov_device_vector_precision: vector handle is null");
        return -1;
    }
    return vec->precision;
}

void* qkrylov_device_vector_data(qkrylov_device_vector_h vec) {
    if (!vec) {
        set_last_error("qkrylov_device_vector_data: vector handle is null");
        return nullptr;
    }
    if (vec->precision == 1) {
        return qkrylov_device_vector_data_fp64(vec);
    } else if (vec->precision == 0) {
        return qkrylov_device_vector_data_fp32(vec);
    } else {
        set_last_error("qkrylov_device_vector_data: unknown precision");
        return nullptr;
    }
}

qkrylov_device_vector_h qkrylov_device_vector_create(uint64_t dim) {
    return qkrylov_device_vector_create_fp64(dim);
}

int qkrylov_device_vector_copy_from_host(qkrylov_device_vector_h dst, const double* host_src_complex) {
    return qkrylov_device_vector_copy_from_host_fp64(dst, host_src_complex);
}

int qkrylov_device_vector_copy_to_host(const qkrylov_device_vector_h src, double* host_dst_complex) {
    return qkrylov_device_vector_copy_to_host_fp64(src, host_dst_complex);
}

int qkrylov_device_vector_dot(const qkrylov_device_vector_h x, const qkrylov_device_vector_h y, double* dot_re, double* dot_im) {
    return qkrylov_device_vector_dot_fp64(x, y, dot_re, dot_im);
}

int qkrylov_device_vector_norm(const qkrylov_device_vector_h x, double* norm_out) {
    return qkrylov_device_vector_norm_fp64(x, norm_out);
}

int qkrylov_device_vector_axpy(double a_re, double a_im, const qkrylov_device_vector_h x, qkrylov_device_vector_h y) {
    return qkrylov_device_vector_axpy_fp64(a_re, a_im, x, y);
}

int qkrylov_device_vector_scal(double a_re, double a_im, qkrylov_device_vector_h x) {
    return qkrylov_device_vector_scal_fp64(a_re, a_im, x);
}

int qkrylov_device_vector_normalize(qkrylov_device_vector_h x) {
    return qkrylov_device_vector_normalize_fp64(x);
}

int qkrylov_device_vector_zero_fill(qkrylov_device_vector_h x) {
    return qkrylov_device_vector_zero_fill_fp64(x);
}

int qkrylov_device_vector_copy(const qkrylov_device_vector_h src, qkrylov_device_vector_h dst) {
    return qkrylov_device_vector_copy_fp64(src, dst);
}

} // extern "C"
