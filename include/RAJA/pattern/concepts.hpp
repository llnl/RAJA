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
    RAJA::type_traits::is_same_decay_v<decltype(Pol::policy), ::RAJA::Policy> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::pattern),
                                       ::RAJA::Pattern> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::launch), ::RAJA::Launch> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::platform),
                                       ::RAJA::Platform>;

template<typename T>
concept IndexSetType =
    static_cast<bool>(RAJA::type_traits::is_index_set<std::decay_t<T>>::value);

template<typename T>
concept IndexSetPolicy =
    static_cast<bool>(type_traits::is_indexset_policy<std::decay_t<T>>::value);

template<typename T>
concept Resource = RAJA::type_traits::is_resource<std::decay_t<T>>::value;

template<typename T>
concept ForallParams =
    expt::type_traits::is_ForallParamPack<std::decay_t<T>>::value;

template<typename T>
concept EmptyForallParams =
    ForallParams<T> &&
    RAJA::expt::type_traits::is_ForallParamPack_empty<T>::value;

template<typename T>
concept NonEmptyForallParams =
    ForallParams<T> &&
    !RAJA::expt::type_traits::is_ForallParamPack_empty<T>::value;

template<typename T>
concept MultiPolicyConcept =
    static_cast<bool>(RAJA::type_traits::is_multi_policy<T>::value);

/// Iteration mappings inherit from things like StridedLoopBase
#define MakeExecPolWithIterMappingConcept(ConceptName, BaseName)               \
  template<typename Pol>                                                       \
  concept ConceptName =                                                        \
      ExecutionPolicy<Pol> &&                                                  \
      std::is_base_of_v<BaseName, typename Pol::IterationMapping>;

// clang-format off
MakeExecPolWithIterMappingConcept(StridedLoopPolicy,
                                  iteration_mapping::StridedLoopBase)
MakeExecPolWithIterMappingConcept(UnsizedLoopPolicy,
                                  iteration_mapping::UnsizedLoopBase)
MakeExecPolWithIterMappingConcept(SizedLoopPolicy,
                                  iteration_mapping::SizedLoopBase)
MakeExecPolWithIterMappingConcept(ContiguousLoopPolicy,
                                  iteration_mapping::ContiguousLoopBase)
MakeExecPolWithIterMappingConcept(DirectPolicy,
                                  iteration_mapping::Direct)
MakeExecPolWithIterMappingConcept(DirectBasePolicy,
                                  iteration_mapping::DirectBase)
// clang-format on
}  // namespace concepts

namespace type_traits
{
DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);

}  // namespace type_traits


}  // namespace RAJA

#endif