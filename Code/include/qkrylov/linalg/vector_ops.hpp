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
    Complex s = 0.0;

    for(std::size_t i=0;i<x.size();++i)
    {
        s += std::conj(x[i]) * y[i];
    }

    return s;
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
    for(auto& v : x)
    {
        v *= a;
    }
}

inline void axpy(
    Complex a,
    const Vector& x,
    Vector& y
)
{
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
