#include "qkrylov/hamiltonian/matrix_free_hamiltonian.hpp"

#include <stdexcept>

namespace qkrylov
{

MatrixFreeHamiltonian::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    std::shared_ptr<Site> site,
    const OpSum& ops
)
    : basis_(std::move(basis)),
      ops_(ops)
{
    const int N = (int)ops_.size() > 0 ? (int)basis_->nsites() : 0;
    // Usually basis->nsites() should give us how many sites we need.
    // If not, we just fill with the provided site.
    sites_.assign(basis_->nsites(), site);
}

MatrixFreeHamiltonian::MatrixFreeHamiltonian(
    std::shared_ptr<Basis> basis,
    const std::vector<std::shared_ptr<Site>>& sites,
    const OpSum& ops
)
    : basis_(std::move(basis)),
      sites_(sites),
      ops_(ops)
{
}

Index MatrixFreeHamiltonian::dimension() const
{
    return basis_->size();
}

void MatrixFreeHamiltonian::apply(
    const Vector& x,
    Vector& y
) const
{
    const Index dim =
        basis_->size();

    y.assign(
        dim,
        Complex(0.0,0.0)
    );

    #pragma omp parallel for
    for(Index alpha=0;
        alpha<dim;
        ++alpha)
    {
        const StateID initial_state =
            basis_->state(alpha);

        const Complex xin =
            x[alpha];

        for(const auto& term :
            ops_.terms())
        {
            StateID state =
                initial_state;

            Complex amp =
                term.coeff;

            bool valid = true;

            for(const auto& factor :
                term.factors)
            {
                // Decompose global state to local state
                StateID temp_state = state;
                for(int i=0; i<factor.site; ++i) temp_state /= sites_[i]->dimension();
                StateID local_state = temp_state % sites_[factor.site]->dimension();

                auto action =
                    sites_.at(factor.site)->apply(
                        factor.op,
                        local_state
                    );

                if(!action.valid)
                {
                    valid = false;
                    break;
                }

                // Global phase handling for fermions (Jordan-Wigner)
                if (sites_[factor.site]->is_fermionic()) {
                    int count = 0;
                    for(int i=0; i<factor.site; ++i) {
                        if (sites_[i]->is_fermionic()) {
                            // Extract local occupancy
                            StateID temp = state;
                            for(int k=0; k<i; ++k) temp /= sites_[k]->dimension();
                            StateID loc = temp % sites_[i]->dimension();
                            // Count particles for phase (assuming bit 0 is occupied for fermion site)
                            if (loc & 1ULL) count++;
                            // For Hubbard, check both bits (Spin-Down is usually bit 1)
                            if (sites_[i]->dimension() == 4 && (loc & 2ULL)) count++;
                        }
                    }
                    if (count % 2 != 0) amp *= -1.0;
                }

                // Re-compose new state
                StateID new_local_state = action.new_state;
                StateID power_of_dim = 1;
                for(int i=0; i<factor.site; ++i) power_of_dim *= sites_[i]->dimension();

                state = (state - local_state * power_of_dim) + new_local_state * power_of_dim;

                amp *=
                    action.matrix_element;
            }

            if(!valid)
                continue;

            if(!basis_->contains(state))
                continue;

            const Index beta =
                basis_->index(state);

            Complex val = amp * xin;
            double* ptr = reinterpret_cast<double*>(&y[beta]);
            #pragma omp atomic
            ptr[0] += val.real();
            #pragma omp atomic
            ptr[1] += val.imag();
        }
    }
}

MatrixFreeHamiltonian::Vector MatrixFreeHamiltonian::diagonal() const
{
    const Index dim = basis_->size();
    Vector diag(dim, Complex(0.0, 0.0));

    for (Index alpha = 0; alpha < dim; ++alpha) {
        const StateID initial_state = basis_->state(alpha);

        for (const auto& term : ops_.terms()) {
            StateID state = initial_state;
            Complex amp = term.coeff;
            bool valid = true;

            for (const auto& factor : term.factors) {
                StateID temp_state = state;
                for(int i=0; i<factor.site; ++i) temp_state /= sites_[i]->dimension();
                StateID local_state = temp_state % sites_[factor.site]->dimension();

                auto action = sites_.at(factor.site)->apply(factor.op, local_state);
                if (!action.valid) {
                    valid = false;
                    break;
                }
                StateID new_local_state = action.new_state;
                StateID power_of_dim = 1;
                for(int i=0; i<factor.site; ++i) power_of_dim *= sites_[i]->dimension();

                state = (state - local_state * power_of_dim) + new_local_state * power_of_dim;
                amp *= action.matrix_element;
            }

            if (valid && state == initial_state) {
                diag[alpha] += amp;
            }
        }
    }
    return diag;
}

}
