#include "qkrylov/core/types.hpp"
#include "qkrylov/solvers/davidson.hpp"

#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <Kokkos_Core.hpp>

namespace qkrylov {
namespace QKRYLOV_PRECISION_NAMESPACE {


namespace
{
struct SmallEigensystem {
    std::vector<Real> eigenvalues;
    std::vector<std::vector<Complex>> eigenvectors;
};

// Jacobi rotation-based eigensolver for small Hermitian matrices
SmallEigensystem solve_small_hermitian(const std::vector<std::vector<Complex>>& H) {
    int n = H.size();
    std::vector<std::vector<Complex>> A = H;
    std::vector<std::vector<Complex>> V(n, std::vector<Complex>(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;

    const Real jacobi_tol = std::numeric_limits<Real>::epsilon() * Real(10.0);

    for (int iter = 0; iter < 1000; ++iter) {
        Real max_off = 0.0;
        int p = 0, q = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A[i][j]) > max_off) {
                    max_off = std::abs(A[i][j]);
                    p = i; q = j;
                }
            }
        }

        if (max_off < jacobi_tol) break;

        Complex app = A[p][p];
        Complex aqq = A[q][q];
        Complex apq = A[p][q];

        Real phi = 0.5 * std::atan2(2.0 * std::abs(apq), std::real(app - aqq));
        Complex c = std::cos(phi);
        Complex s = std::sin(phi) * apq / std::abs(apq);

        for (int i = 0; i < n; ++i) {
            Complex vip = V[i][p];
            Complex viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -std::conj(s) * vip + c * viq;

            if (i != p && i != q) {
                Complex aip = A[i][p];
                Complex aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip + s * aiq;
                A[i][q] = A[q][i] = -std::conj(s) * aip + c * aiq;
            }
        }
        A[p][p] = c * c * app + s * std::conj(s) * aqq + Real(2.0) * std::real(c * s * std::conj(apq));
        A[q][q] = s * std::conj(s) * app + c * c * aqq - Real(2.0) * std::real(c * s * std::conj(apq));
        A[p][q] = A[q][p] = Real(0.0);
    }

    SmallEigensystem res;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return std::real(A[i][i]) < std::real(A[j][j]);
    });

    for (int i : idx) {
        res.eigenvalues.push_back(std::real(A[i][i]));
        std::vector<Complex> v(n);
        for (int j = 0; j < n; ++j) v[j] = V[j][i];
        res.eigenvectors.push_back(v);
    }
    return res;
}
}

template <typename ExecSpace>
DavidsonResult davidson_lowest(
    const MatrixFreeHamiltonian<ExecSpace>& H,
    int n_eig,
    int max_subspace,
    Real tol
)
{
    const Index dim = H.dimension();
    if (dim == 0) return {};

    n_eig = std::min<int>(n_eig, dim);
    max_subspace = std::max(max_subspace, 2 * n_eig);

    auto diag_dev = H.diagonal();
    auto diag_host = Kokkos::create_mirror_view(diag_dev);
    Kokkos::deep_copy(diag_host, diag_dev);

    std::vector<VectorView<ExecSpace>> V;
    std::vector<VectorView<ExecSpace>> HV;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    // Initial subspace: use some diagonal dominance if possible,
    // or just unit vectors for the smallest diagonal elements
    std::vector<Index> diag_idx(dim);
    for(Index i=0; i<dim; ++i) diag_idx[i] = i;
    std::sort(diag_idx.begin(), diag_idx.end(), [&](Index i, Index j){
        return diag_host(i).real() < diag_host(j).real();
    });

    const Real lin_indep_tol = std::numeric_limits<Real>::epsilon() * Real(100.0);

    for (int i = 0; i < n_eig; ++i) {
        VectorView<ExecSpace> v("v", dim);
        auto v_host = Kokkos::create_mirror_view(v);
        v_host(diag_idx[i]) = 1.0;
        Kokkos::deep_copy(v, v_host);
        
        // Orthogonalize against previous V
        for (const auto& v_prev : V) axpy(-dot(v_prev, v), v_prev, v);
        Real nrm = norm(v);
        if (nrm < lin_indep_tol) {
            // Fallback to random if unit vector is not linearly independent
            for (Index j = 0; j < dim; ++j) v_host(j) = KComplex(dist(rng), dist(rng));
            Kokkos::deep_copy(v, v_host);
            for (const auto& v_prev : V) axpy(-dot(v_prev, v), v_prev, v);
            nrm = norm(v);
        }
        scal(1.0/nrm, v);
        V.push_back(v);
        
        VectorView<ExecSpace> hv("hv", dim);
        H.apply(v, hv);
        HV.push_back(hv);
    }

    const Real denom_cutoff = std::numeric_limits<Real>::epsilon() * Real(100.0);

    for (int iter = 0; iter < 100; ++iter) {
        int m = V.size();
        std::vector<std::vector<Complex>> H_sub(m, std::vector<Complex>(m));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i; ++j) {
                KComplex val = dot(V[i], HV[j]);
                H_sub[i][j] = Complex(val.real(), val.imag());
                if (i != j) H_sub[j][i] = std::conj(H_sub[i][j]);
            }
        }

        auto eig = solve_small_hermitian(H_sub);

        bool all_converged = true;
        std::vector<VectorView<ExecSpace>> next_corrections;

        for (int k = 0; k < n_eig; ++k) {
            Real lambda = eig.eigenvalues[k];
            const auto& s = eig.eigenvectors[k];

            VectorView<ExecSpace> r("r", dim);
            for (int i = 0; i < m; ++i) {
                axpy(KComplex(s[i].real(), s[i].imag()), HV[i], r);
                axpy(KComplex(-lambda * s[i].real(), -lambda * s[i].imag()), V[i], r);
            }

            Real res_norm = norm(r);
            if (res_norm > tol) {
                all_converged = false;
                VectorView<ExecSpace> t("t", dim);
                Kokkos::parallel_for("davidson_precond", dim, KOKKOS_LAMBDA(const Index i) {
                    KComplex diff = diag_dev(i) - lambda;
                    if (Kokkos::abs(diff) < denom_cutoff) diff = (diff.real() >= Real(0.0)) ? denom_cutoff : -denom_cutoff;
                    t(i) = r(i) / diff;
                });
                next_corrections.push_back(t);
            }
        }

        if (all_converged || iter == 99) {
            DavidsonResult res;
            res.iterations = iter + 1;
            res.converged = all_converged;
            for (int k = 0; k < n_eig; ++k) {
                res.eigenvalues.push_back(eig.eigenvalues[k]);
                VectorView<ExecSpace> ev("ev", dim);
                for (int i = 0; i < m; ++i) axpy(KComplex(eig.eigenvectors[k][i].real(), eig.eigenvectors[k][i].imag()), V[i], ev);
                HostVector host_ev;
                copy_device_to_host(ev, host_ev);
                res.eigenvectors.push_back(host_ev);
            }
            return res;
        }

        // Expansion and restart
        if (V.size() + next_corrections.size() > (size_t)max_subspace) {
            std::vector<VectorView<ExecSpace>> next_V;
            std::vector<VectorView<ExecSpace>> next_HV;
            for (int k = 0; k < n_eig; ++k) {
                VectorView<ExecSpace> ritz_v("ritz_v", dim);
                VectorView<ExecSpace> ritz_hv("ritz_hv", dim);
                for (int i = 0; i < m; ++i) {
                    axpy(KComplex(eig.eigenvectors[k][i].real(), eig.eigenvectors[k][i].imag()), V[i], ritz_v);
                    axpy(KComplex(eig.eigenvectors[k][i].real(), eig.eigenvectors[k][i].imag()), HV[i], ritz_hv);
                }
                next_V.push_back(ritz_v);
                next_HV.push_back(ritz_hv);
            }
            V = next_V;
            HV = next_HV;
        }

        for (auto& t : next_corrections) {
            for (const auto& v : V) axpy(-dot(v, t), v, t);
            Real nrm = norm(t);
            if (nrm > lin_indep_tol) {
                scal(1.0/nrm, t);
                V.push_back(t);
                VectorView<ExecSpace> ht("ht", dim);
                H.apply(t, ht);
                HV.push_back(ht);
            }
        }
    }

    return {};
}


// Explicit instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template DavidsonResult davidson_lowest<Kokkos::Serial>(const MatrixFreeHamiltonian<Kokkos::Serial>&, int, int, Real);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template DavidsonResult davidson_lowest<Kokkos::OpenMP>(const MatrixFreeHamiltonian<Kokkos::OpenMP>&, int, int, Real);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template DavidsonResult davidson_lowest<Kokkos::Threads>(const MatrixFreeHamiltonian<Kokkos::Threads>&, int, int, Real);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template DavidsonResult davidson_lowest<Kokkos::Cuda>(const MatrixFreeHamiltonian<Kokkos::Cuda>&, int, int, Real);
#endif
#ifdef KOKKOS_ENABLE_HIP
template DavidsonResult davidson_lowest<Kokkos::HIP>(const MatrixFreeHamiltonian<Kokkos::HIP>&, int, int, Real);
#endif
#ifdef KOKKOS_ENABLE_SYCL
template DavidsonResult davidson_lowest<Kokkos::Experimental::SYCL>(const MatrixFreeHamiltonian<Kokkos::Experimental::SYCL>&, int, int, Real);
#endif
}

}
