#include "qkrylov/solvers/lanczos.hpp"

#include <random>
#include <stdexcept>

namespace qkrylov
{

LanczosResult lanczos_ground_state(
    const MatrixFreeHamiltonian& H,
    int maxiter,
    double tol
)
{
    const Index dim =
        H.dimension();

    Vector v0(
        dim,
        Complex(0.0,0.0)
    );

    Vector v1(
        dim,
        Complex(0.0,0.0)
    );

    Vector w(
        dim,
        Complex(0.0,0.0)
    );

    std::mt19937 rng(1234);

    std::uniform_real_distribution<double>
        dist(-1.0,1.0);

    for(Index i=0;i<dim;++i)
    {
        v1[i] = dist(rng);
    }

    normalize(v1);

    double alpha = 0.0;
    double beta  = 0.0;

    double energy_old =
        1.0e100;

    double energy =
        0.0;

    for(int iter=0;
        iter<maxiter;
        ++iter)
    {
        H.apply(v1,w);

        alpha =
            std::real(
                dot(v1,w)
            );

        axpy(
            -alpha,
            v1,
            w
        );

        if(iter>0)
        {
            axpy(
                -beta,
                v0,
                w
            );
        }

        beta =
            norm(w);

        energy = alpha;

        if(std::abs(
            energy-energy_old
        ) < tol)
        {
            break;
        }

        energy_old =
            energy;

        v0 = v1;

        if(beta > 0.0)
        {
            v1 = w;

            scal(
                1.0/beta,
                v1
            );
        }
    }

    LanczosResult res;

    res.energy =
        energy;

    res.eigenvector =
        v1;

    return res;
}

}
