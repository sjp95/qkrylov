#pragma once

#include <type_traits>

namespace qkrylov {
namespace solvers {
namespace policy {

// Clean Policy Tags (Modern C++20, no legacy baggage)
struct OnePass {};       // Default: Ground state energy only, 3 vectors in RAM, no DGKS
struct OnePass_DKGS {};  // Arbitrary number of low energy states + exact vectors: m vectors in RAM, DGKS enabled
struct OnePass_full {};  // Ground state only + vector: m vectors in RAM, no DGKS
struct TwoPass {};       // Ground state only + vector: 4 vectors in RAM, seed replay

using Default      = OnePass;
using OnePass_DGKS = OnePass_DKGS; // Spelling convenience alias

// Policy Traits
template <typename Policy>
struct policy_traits;

template <>
struct policy_traits<OnePass> {
    static constexpr bool supports_multistate = false; // Ground state only
    static constexpr bool stores_basis        = false;
    static constexpr bool uses_dgks           = false;
    static constexpr bool computes_vector     = false; // No eigenvector allocation
    static constexpr bool is_two_pass         = false;
};

template <>
struct policy_traits<OnePass_DKGS> {
    static constexpr bool supports_multistate = true;  // Arbitrary number of low energy states
    static constexpr bool stores_basis        = true;
    static constexpr bool uses_dgks           = true;  // 2-pass DGKS reorthogonalization
    static constexpr bool computes_vector     = true;  // Exact eigenvector return
    static constexpr bool is_two_pass         = false;
};

template <>
struct policy_traits<OnePass_full> {
    static constexpr bool supports_multistate = false; // Ground state only
    static constexpr bool stores_basis        = true;
    static constexpr bool uses_dgks           = false;
    static constexpr bool computes_vector     = true;  // Ground state eigenvector return
    static constexpr bool is_two_pass         = false;
};

template <>
struct policy_traits<TwoPass> {
    static constexpr bool supports_multistate = false; // Ground state only
    static constexpr bool stores_basis        = false;
    static constexpr bool uses_dgks           = false;
    static constexpr bool computes_vector     = true;  // Ground state eigenvector return
    static constexpr bool is_two_pass         = true;  // Deterministic seed replay
};

// Type Validation
template <typename T>
struct is_policy : std::false_type {};

template <> struct is_policy<OnePass>      : std::true_type {};
template <> struct is_policy<OnePass_DKGS> : std::true_type {};
template <> struct is_policy<OnePass_full> : std::true_type {};
template <> struct is_policy<TwoPass>      : std::true_type {};

template <typename T>
inline constexpr bool is_policy_v = is_policy<T>::value;

} // namespace policy
} // namespace solvers
} // namespace qkrylov
