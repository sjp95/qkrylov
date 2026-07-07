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
      site_(std::move(site)),
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
                auto action =
                    site_->apply(
                        factor.op,
                        factor.site,
                        state
                    );

                if(!action.valid)
                {
                    valid = false;
                    break;
                }

                state =
                    action.new_state;

                amp *=
                    action.matrix_element;
            }

            if(!valid)
                continue;

            if(!basis_->contains(state))
                continue;

            const Index beta =
                basis_->index(state);

            y[beta] +=
                amp*xin;
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
                auto action = site_->apply(factor.op, factor.site, state);
                if (!action.valid) {
                    valid = false;
                    break;
                }
                state = action.new_state;
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
