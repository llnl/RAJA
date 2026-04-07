/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA message queue policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_msg_queue_HPP
#define policy_msg_queue_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace messages
{

///
/// This is a view-like queue so that message queues can be copied to kernels.
///
template<typename Container, typename Policy, typename Args>
class queue;

}  // namespace messages

namespace policy
{
namespace messages
{

//
//////////////////////////////////////////////////////////////////////
//
// Queue policies
//
//////////////////////////////////////////////////////////////////////
//

template<bool Overwrite = false>
struct mpsc_queue
{
  static constexpr bool should_overwrite = Overwrite;
};

template<bool Overwrite = false>
struct spsc_queue
{
  static constexpr bool should_overwrite = Overwrite;
};

}  // namespace messages
}  // namespace policy

using spsc_queue = policy::messages::spsc_queue<false>;
using mpsc_queue = policy::messages::mpsc_queue<false>;

}  // namespace RAJA

#endif
