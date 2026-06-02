#include "qkrylov/operators/opsum.hpp"

namespace qkrylov
{

void OpSum::add_term(
    const OperatorTerm& term
)
{
    terms_.push_back(term);
}

void OpSum::clear()
{
    terms_.clear();
}

std::size_t OpSum::size() const
{
    return terms_.size();
}

const OperatorTerm&
OpSum::operator[](
    std::size_t i
) const
{
    return terms_[i];
}

const std::vector<OperatorTerm>&
OpSum::terms() const
{
    return terms_;
}

}