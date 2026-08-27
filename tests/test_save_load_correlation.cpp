#include "qkrylov/solvers/lanczos.hpp"
#include "qkrylov/basis/spinhalf_basis.hpp"
#include "qkrylov/sites/spinhalf_site.hpp"
#include "qkrylov/operators/opsum.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

using namespace qkrylov;

// Helper function to save energy and eigenvector to binary file
void save_state_binary(const std::string& filename, double energy, const Vector& vec) {
    std::ofstream out(filename, std::ios::binary);
    assert(out.is_open());

    out.write(reinterpret_cast<const char*>(&energy), sizeof(double));
    uint64_t size = vec.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(Complex));
    out.close();
}

// Helper function to load energy and eigenvector from binary file
void load_state_binary(const std::string& filename, double& energy, Vector& vec) {
    std::ifstream in(filename, std::ios::binary);
    assert(in.is_open());

    in.read(reinterpret_cast<char*>(&energy), sizeof(double));
    uint64_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(Complex));
    in.close();
}

int main() {
    std::cout << "Running C++ save/load state & correlation calculation test..." << std::endl;

    int N = 4;
    auto basis = std::make_shared<SpinHalfBasis>(N);
    auto site = std::make_shared<SpinHalfSite>();

    // 1. Solve for ground state of 4-site Heisenberg chain
    OpSum os_h;
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        os_h.add_term({1.0, {{"Sz", i}, {"Sz", j}}});
        os_h.add_term({0.5, {{"Sp", i}, {"Sm", j}}});
        os_h.add_term({0.5, {{"Sm", i}, {"Sp", j}}});
    }

    MatrixFreeHamiltonian H(basis, site, os_h);
    auto l_res = lanczos_ground_state(H, 100, 1e-12);

    // 2. Save energy and eigenvector to file
    std::string filename = "ground_state_c.bin";
    save_state_binary(filename, l_res.energy, l_res.eigenvector);
    std::cout << "Saved state to " << filename << " with energy " << l_res.energy << std::endl;

    // 3. Load state back in a separate step
    double loaded_energy = 0.0;
    Vector loaded_psi0;
    load_state_binary(filename, loaded_energy, loaded_psi0);
    std::cout << "Loaded state from " << filename << " with energy " << loaded_energy << std::endl;

    assert(std::abs(loaded_energy - l_res.energy) < 1e-12);
    assert(loaded_psi0.size() == l_res.eigenvector.size());

    // 4. Reuse loaded eigenvector to compute spin-spin correlation <psi_0 | Sz_0 Sz_2 | psi_0>
    OpSum os_corr;
    os_corr.add_term({1.0, {{"Sz", 0}, {"Sz", 2}}});
    MatrixFreeHamiltonian Sz0Sz2(basis, site, os_corr);

    Vector A_psi0(H.dimension(), 0.0);
    Sz0Sz2.apply(loaded_psi0.data(), A_psi0.data());

    Complex corr_val = dot(loaded_psi0, A_psi0);
    std::cout << "<Sz_0 Sz_2> correlation value = " << corr_val.real() << std::endl;

    // In 4-site Heisenberg ring ground state, <Sz_0 Sz_2> is positive (next-nearest neighbor)
    assert(std::abs(corr_val.imag()) < 1e-10);
    assert(corr_val.real() > 0.0);

    std::cout << "C++ save/load & correlation calculation passed successfully!" << std::endl;
    return 0;
}
