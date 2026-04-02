/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA sequential policy definitions.
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

#ifndef policy_sequential_HPP
#define policy_sequential_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace sequential
{

enum struct multi_reduce_algorithm : int
{
  left_fold
};

template<multi_reduce_algorithm t_multi_algorithm>
struct MultiReduceTuning
{
  static constexpr multi_reduce_algorithm algorithm = t_multi_algorithm;
  static constexpr bool consistent =
      (algorithm == multi_reduce_algorithm::left_fold);
};

}  // namespace sequential

namespace policy
{
namespace sequential
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

struct seq_region : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                          Pattern::region,
                                                          Launch::sync,
                                                          Platform::host>
{};

struct seq_launch_t : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                            Pattern::region,
                                                            Launch::sync,
                                                            Platform::host>
{};

struct seq_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                        Pattern::forall,
                                                        Launch::undefined,
                                                        Platform::host>
{};

///
/// Index set segment iteration policies
///
using seq_segit = seq_exec;

///
/// WorkGroup execution policies
///
struct seq_work : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                        Pattern::workgroup_exec,
                                                        Launch::sync,
                                                        Platform::host>
{};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct seq_reduce : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                          Pattern::reduce,
                                                          Launch::undefined,
                                                          Platform::host>
{};

///
template<typename tuning>
struct seq_multi_reduce_policy : make_policy_pattern_launch_platform_t<
                                     Policy::sequential,
                                     Pattern::multi_reduce,
                                     Launch::undefined,
                                     Platform::host,
                                     std::conditional_t<tuning::consistent,
                                                        reduce::ordered,
                                                        reduce::unordered>>
{};

///
///////////////////////////////////////////////////////////////////////
///
/// Atomic execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct seq_atomic
{};

struct seq_thread
{};

template<RAJA::sequential::multi_reduce_algorithm algorithm>
using seq_multi_reduce_tuning =
    seq_multi_reduce_policy<RAJA::sequential::MultiReduceTuning<algorithm>>;

// Policies for RAJA::MultiReduce* objects with specific behaviors.
// - left_fold policies combine new values into a single value.
using seq_multi_reduce_left_fold = seq_multi_reduce_tuning<
    RAJA::sequential::multi_reduce_algorithm::left_fold>;

// Policy for RAJA::MultiReduce* objects that gives the
// same answer every time when used in the same way
using seq_multi_reduce = seq_multi_reduce_left_fold;

}  // namespace sequential
}  // namespace policy

using policy::sequential::seq_atomic;
using policy::sequential::seq_exec;
using policy::sequential::seq_launch_t;
using policy::sequential::seq_multi_reduce;
using policy::sequential::seq_reduce;
using policy::sequential::seq_region;
using policy::sequential::seq_segit;
using policy::sequential::seq_thread;
using policy::sequential::seq_work;


}  // namespace RAJA

#endif
