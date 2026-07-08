#pragma once

#include "qkrylov/core/types.hpp"

#include <cmath>
#include <vector>

namespace qkrylov
{

using Vector =
    std::vector<Complex>;

inline Complex dot(
    const Vector& x,
    const Vector& y
)
{
    double re = 0.0;
    double im = 0.0;

    #pragma omp parallel for reduction(+:re, im)
    for(std::size_t i=0;i<x.size();++i)
    {
        Complex val = std::conj(x[i]) * y[i];
        re += val.real();
        im += val.imag();
    }

    return Complex(re, im);
}

inline double norm(
    const Vector& x
)
{
    return std::sqrt(
        std::real(
            dot(x,x)
        )
    );
}

inline void scal(
    Complex a,
    Vector& x
)
{
    #pragma omp parallel for
    for(std::size_t i=0; i<x.size(); ++i)
    {
        x[i] *= a;
    }
}

inline void axpy(
    Complex a,
    const Vector& x,
    Vector& y
)
{
    #pragma omp parallel for
    for(std::size_t i=0;i<x.size();++i)
    {
        y[i] += a*x[i];
    }
}

inline void normalize(
    Vector& x
)
{
    const double n =
        norm(x);

    if(n==0.0)
        return;

    for(auto& v : x)
    {
        v /= n;
    }
}

}
