#pragma once

#include "operator_term.hpp"

#include <vector>

namespace qkrylov
{

class OpSum
{
public:

    void add_term(
        const OperatorTerm& term
    );

    OpSum& operator+=(const OperatorTerm& term)
    {
        add_term(term);
        return *this;
    }

    void clear();

    std::size_t size() const;

    const OperatorTerm&
    operator[](std::size_t i) const;

    const std::vector<OperatorTerm>&
    terms() const;

private:

    std::vector<OperatorTerm> terms_;
};

}