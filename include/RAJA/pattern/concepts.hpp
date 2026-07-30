/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining core C++ concepts for use in development of
 *RAJA
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_type_concepts_HPP
#define RAJA_type_concepts_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/detail/TypeTraits.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/index/IndexSet.hpp"

namespace RAJA
{

namespace concepts
{

/// A RAJA ExecutionPolicy is a backend-specific directive supplied by the user
/// like hip_exec or seq_exec that instructs RAJA how to configure a parallel
/// kernel
template<typename Pol>
concept ExecutionPolicy =
    type_traits::is_same_decay_v<decltype(std::decay_t<Pol>::policy),
                                 ::RAJA::Policy> &&
    type_traits::is_same_decay_v<decltype(std::decay_t<Pol>::pattern),
                                 ::RAJA::Pattern> &&
    type_traits::is_same_decay_v<decltype(std::decay_t<Pol>::launch),
                                 ::RAJA::Launch> &&
    type_traits::is_same_decay_v<decltype(std::decay_t<Pol>::platform),
                                 ::RAJA::Platform>;

/// Note: IndexSetType, IndexSetPolicy, and MultiPolicyConcept all utilize
/// specializations of camp::num<bool=false/true>. Because of this, their
/// value type is actually const long, not bool.  Therefore, static_cast to
/// bool is used below to define these.
template<typename T>
concept IndexSetType =
    static_cast<bool>(type_traits::is_index_set<std::decay_t<T>>::value);

template<typename T>
concept IndexSetPolicy =
    static_cast<bool>(type_traits::is_indexset_policy<std::decay_t<T>>::value);

template<typename T>
concept MultiPolicyConcept =
    static_cast<bool>(type_traits::is_multi_policy<std::decay_t<T>>::value);

template<typename T>
concept Resource = type_traits::is_resource<std::decay_t<T>>::value;

template<typename T>
concept ForallParams =
    expt::type_traits::is_ForallParamPack<std::decay_t<T>>::value;

template<typename T>
concept EmptyForallParams =
    ForallParams<T> &&
    expt::type_traits::is_ForallParamPack_empty<std::decay_t<T>>::value;

template<typename T>
concept NonEmptyForallParams =
    ForallParams<T> &&
    !expt::type_traits::is_ForallParamPack_empty<std::decay_t<T>>::value;

/// This macro creates ExecutionPolicy concepts whose IterationMapping member
/// inherit from a specific type of loop mapping: like a StridedLoop, for
/// example
#define RAJAMakeExecPolWithIterMappingConcept(ConceptName, BaseName)           \
  template<typename Pol>                                                       \
  concept ConceptName =                                                        \
      ExecutionPolicy<Pol> &&                                                  \
      std::is_base_of_v<BaseName,                                              \
                        typename std::decay_t<Pol>::IterationMapping>;

// clang-format off
RAJAMakeExecPolWithIterMappingConcept(StridedLoopPolicy,
                                      iteration_mapping::StridedLoopBase)
RAJAMakeExecPolWithIterMappingConcept(UnsizedLoopPolicy,
                                      iteration_mapping::UnsizedLoopBase)
RAJAMakeExecPolWithIterMappingConcept(SizedLoopPolicy,
                                      iteration_mapping::SizedLoopBase)
RAJAMakeExecPolWithIterMappingConcept(ContiguousLoopPolicy,
                                      iteration_mapping::ContiguousLoopBase)
RAJAMakeExecPolWithIterMappingConcept(DirectPolicy,
                                      iteration_mapping::Direct)
RAJAMakeExecPolWithIterMappingConcept(DirectBasePolicy,
                                      iteration_mapping::DirectBase)
// clang-format on
}  // namespace concepts

namespace type_traits
{
template<typename T>
struct is_execution_policy : std::bool_constant<concepts::ExecutionPolicy<T>>
{};

template<typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

}  // namespace type_traits


}  // namespace RAJA

#endif
