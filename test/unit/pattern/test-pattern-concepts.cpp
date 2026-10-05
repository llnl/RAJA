//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for RAJA pattern concepts.
///

#include "RAJA/pattern/concepts.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/pattern/params/forall.hpp"
#include "RAJA/pattern/params/kernel_name.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA_gtest.hpp"

namespace
{

using TestIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment>;
using TestIndexSetPolicy = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
using TestEmptyForallParams =
    decltype(RAJA::expt::make_forall_param_pack([] {}));
using TestNonEmptyForallParams = decltype(
    RAJA::expt::make_forall_param_pack(RAJA::Name("pattern-concepts"), [] {}));
using TestMultiPolicy =
    RAJA::MultiPolicy<int (*)(RAJA::RangeSegment const&), RAJA::seq_exec>;

struct DummyDirectPolicy
    : RAJA::make_policy_pattern_launch_platform_t<RAJA::Policy::sequential,
                                                  RAJA::Pattern::forall,
                                                  RAJA::Launch::sync,
                                                  RAJA::Platform::host>
{
  using IterationMapping = RAJA::iteration_mapping::Direct;
};

struct DummyStridedUnsizedPolicy
    : RAJA::make_policy_pattern_launch_platform_t<RAJA::Policy::sequential,
                                                  RAJA::Pattern::forall,
                                                  RAJA::Launch::sync,
                                                  RAJA::Platform::host>
{
  using IterationMapping =
      RAJA::iteration_mapping::StridedLoop<RAJA::named_usage::unspecified>;
};

struct DummyContiguousSizedPolicy
    : RAJA::make_policy_pattern_launch_platform_t<RAJA::Policy::sequential,
                                                  RAJA::Pattern::forall,
                                                  RAJA::Launch::sync,
                                                  RAJA::Platform::host>
{
  using IterationMapping = RAJA::iteration_mapping::Contiguousloop<16>;
};

template<typename T>
inline constexpr bool execution_policy_v = RAJA::concepts::ExecutionPolicy<T>;

template<typename T>
inline constexpr bool index_set_type_v = RAJA::concepts::IndexSetType<T>;

template<typename T>
inline constexpr bool index_set_policy_v = RAJA::concepts::IndexSetPolicy<T>;

template<typename T>
inline constexpr bool resource_v = RAJA::concepts::Resource<T>;

template<typename T>
inline constexpr bool forall_params_v = RAJA::concepts::ForallParams<T>;

template<typename T>
inline constexpr bool empty_forall_params_v =
    RAJA::concepts::EmptyForallParams<T>;

template<typename T>
inline constexpr bool non_empty_forall_params_v =
    RAJA::concepts::NonEmptyForallParams<T>;

template<typename T>
inline constexpr bool multi_policy_v = RAJA::concepts::MultiPolicyConcept<T>;

template<typename T>
inline constexpr bool strided_loop_policy_v =
    RAJA::concepts::StridedLoopPolicy<T>;

template<typename T>
inline constexpr bool unsized_loop_policy_v =
    RAJA::concepts::UnsizedLoopPolicy<T>;

template<typename T>
inline constexpr bool sized_loop_policy_v = RAJA::concepts::SizedLoopPolicy<T>;

template<typename T>
inline constexpr bool contiguous_loop_policy_v =
    RAJA::concepts::ContiguousLoopPolicy<T>;

template<typename T>
inline constexpr bool direct_policy_v = RAJA::concepts::DirectPolicy<T>;

template<typename T>
inline constexpr bool direct_base_policy_v =
    RAJA::concepts::DirectBasePolicy<T>;

#define RAJA_STATIC_UNARY_CONCEPT_TEST(ConceptVar, ValidType, InvalidType) \
  static_assert(ConceptVar<ValidType>, #ConceptVar " should accept " #ValidType); \
  static_assert(ConceptVar<ValidType&>,                                      \
                #ConceptVar " should accept lvalue references to " #ValidType); \
  static_assert(!ConceptVar<InvalidType>,                                   \
                #ConceptVar " should reject " #InvalidType)

RAJA_STATIC_UNARY_CONCEPT_TEST(execution_policy_v, RAJA::seq_exec, int);
RAJA_STATIC_UNARY_CONCEPT_TEST(index_set_type_v, TestIndexSet, int);
RAJA_STATIC_UNARY_CONCEPT_TEST(index_set_policy_v, TestIndexSetPolicy,
                               RAJA::seq_exec);
RAJA_STATIC_UNARY_CONCEPT_TEST(resource_v, RAJA::resources::Host, int);
RAJA_STATIC_UNARY_CONCEPT_TEST(forall_params_v, TestEmptyForallParams, int);
RAJA_STATIC_UNARY_CONCEPT_TEST(multi_policy_v, TestMultiPolicy, RAJA::seq_exec);
RAJA_STATIC_UNARY_CONCEPT_TEST(strided_loop_policy_v, DummyStridedUnsizedPolicy,
                               DummyDirectPolicy);
RAJA_STATIC_UNARY_CONCEPT_TEST(unsized_loop_policy_v, DummyStridedUnsizedPolicy,
                               DummyDirectPolicy);
RAJA_STATIC_UNARY_CONCEPT_TEST(sized_loop_policy_v, DummyContiguousSizedPolicy,
                               DummyStridedUnsizedPolicy);
RAJA_STATIC_UNARY_CONCEPT_TEST(contiguous_loop_policy_v,
                               DummyContiguousSizedPolicy,
                               DummyStridedUnsizedPolicy);
RAJA_STATIC_UNARY_CONCEPT_TEST(direct_policy_v, DummyDirectPolicy,
                               DummyStridedUnsizedPolicy);
RAJA_STATIC_UNARY_CONCEPT_TEST(direct_base_policy_v, DummyDirectPolicy,
                               DummyStridedUnsizedPolicy);

static_assert(empty_forall_params_v<TestEmptyForallParams>,
              "EmptyForallParams should accept an empty parameter pack");
static_assert(empty_forall_params_v<TestEmptyForallParams&>,
              "EmptyForallParams should accept lvalue references");
static_assert(!empty_forall_params_v<TestNonEmptyForallParams>,
              "EmptyForallParams should reject a non-empty parameter pack");
static_assert(!empty_forall_params_v<int>,
              "EmptyForallParams should reject non-parameter-pack types");

static_assert(non_empty_forall_params_v<TestNonEmptyForallParams>,
              "NonEmptyForallParams should accept a non-empty parameter pack");
static_assert(non_empty_forall_params_v<TestNonEmptyForallParams&>,
              "NonEmptyForallParams should accept lvalue references");
static_assert(!non_empty_forall_params_v<TestEmptyForallParams>,
              "NonEmptyForallParams should reject an empty parameter pack");
static_assert(!non_empty_forall_params_v<int>,
              "NonEmptyForallParams should reject non-parameter-pack types");

#undef RAJA_STATIC_UNARY_CONCEPT_TEST

}  // namespace

TEST(PatternConcepts, compile_time_coverage) { SUCCEED(); }
