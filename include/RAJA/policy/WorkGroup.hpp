/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper for "multi-policy" and dynamic policy selection
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

#ifndef RAJA_Policy_WorkGroup_HPP
#define RAJA_Policy_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "RAJA/util/concepts.hpp"

namespace RAJA
{

namespace policy
{
namespace workgroup
{

/// execute the enqueued loops in the order they were enqueued
/// Note this is intended for debugging, the WorkGroup abstraction is intended
/// to allow running loops in an unordered fashion (loop fusion)
struct ordered
    : RAJA::make_policy_pattern_t<Policy::undefined, Pattern::workgroup_order>
{};

/// execute the enqueued loops in the reverse order from the order that they
/// were enqueued
/// Note this is intended for debugging, the WorkGroup abstraction is intended
/// to allow running loops in an unordered fashion (loop fusion)
struct reverse_ordered
    : RAJA::make_policy_pattern_t<Policy::undefined, Pattern::workgroup_order>
{};

/// store an array of pointers to the enqueued objects. The enqueued objects
/// are stored in separate allocations.
struct array_of_pointers
    : RAJA::make_policy_pattern_t<Policy::undefined, Pattern::workgroup_storage>
{};

/// store an array of pointers to the enqueued objects. The enqueued objects
/// are stored in a single compact array.
struct ragged_array_of_objects
    : RAJA::make_policy_pattern_t<Policy::undefined, Pattern::workgroup_storage>
{};

/// store an array of the enqueued objects with padding such that the objects
/// can be accessed using a constant stride from the beginning of the array.
struct constant_stride_array_of_objects
    : RAJA::make_policy_pattern_t<Policy::undefined, Pattern::workgroup_storage>
{};

/// Dispatch using function pointers to make indirect function calls
struct indirect_function_call_dispatch
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_dispatch>
{};

/// Dispatch using virtual functions to make indirect function calls
struct indirect_virtual_function_dispatch
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_dispatch>
{};

/// Dispatch using an implementation equivalent to a switch statement to select
/// the type from RangeAndCallables and directly call the object.
/// RangeAndCallables is a pack of types of the form camp::list<Range, Callable>
/// where pairs of Range and Callable are the types of the range and callable
/// objects that may be passed to WorkPool enqueue.
template<typename... RangeAndCallables>
struct direct_dispatch
    : RAJA::make_policy_pattern_t<Policy::undefined,
                                  Pattern::workgroup_dispatch>
{};

template<typename EXEC_POLICY_T,
         typename ORDER_POLICY_T,
         typename STORAGE_POLICY_T,
         typename DISPATCH_POLICY_T = indirect_function_call_dispatch>
struct WorkGroupPolicy : public RAJA::make_policy_pattern_platform_t<
                             policy_of<EXEC_POLICY_T>::value,
                             Pattern::workgroup,
                             platform_of<EXEC_POLICY_T>::value>
{
  static_assert(
      RAJA::pattern_is<EXEC_POLICY_T, RAJA::Pattern::workgroup_exec>::value,
      "WorkGroupPolicy: EXEC_POLICY_T must be a workgroup exec policy");
  static_assert(
      RAJA::pattern_is<ORDER_POLICY_T, RAJA::Pattern::workgroup_order>::value,
      "WorkGroupPolicy: ORDER_POLICY_T must be a workgroup order policy");
  static_assert(
      RAJA::pattern_is<STORAGE_POLICY_T,
                       RAJA::Pattern::workgroup_storage>::value,
      "WorkGroupPolicy: STORAGE_POLICY_T must be a workgroup storage policy");
  static_assert(
      RAJA::pattern_is<DISPATCH_POLICY_T,
                       RAJA::Pattern::workgroup_dispatch>::value,
      "WorkGroupPolicy: DISPATCH_POLICY_T must be a workgroup dispatch policy");
};

}  // end namespace workgroup
}  // end namespace policy

using policy::workgroup::ordered;
using policy::workgroup::reverse_ordered;

using policy::workgroup::array_of_pointers;
using policy::workgroup::constant_stride_array_of_objects;
using policy::workgroup::ragged_array_of_objects;

using policy::workgroup::direct_dispatch;
using policy::workgroup::indirect_function_call_dispatch;
using policy::workgroup::indirect_virtual_function_dispatch;

using policy::workgroup::WorkGroupPolicy;

}  // end namespace RAJA

#endif
