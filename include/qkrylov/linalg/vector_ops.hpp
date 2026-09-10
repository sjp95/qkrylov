#pragma once

#include "qkrylov/core/types.hpp"

#include <vector>
#include <cmath>

namespace qkrylov
{

using Vector =
    std::vector<Complex>;

inline Complex dot(
    const Vector& x,
    const Vector& y
)
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(x.size());
    double re = 0.0;
    double im = 0.0;

#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:re, im)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
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
    return std::sqrt(std::real(dot(x, x)));
}

inline void axpy(
    Complex a,
    const Vector& x,
    Vector& y
)
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(x.size());

#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

inline void scal(
    Complex a,
    Vector& x
)
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(x.size());

#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

inline void normalize(
    Vector& x
)
{
    double nrm = norm(x);

    if (nrm > 1.0e-15) {
        scal(1.0 / nrm, x);
    }
}

}
