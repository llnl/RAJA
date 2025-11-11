/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining additional helpers for RAJA::messages
 *          to be properly aligned.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJA_MSG_ALIGN_HPP
#define RAJA_MSG_ALIGN_HPP

#include <cstddef>
#include "camp/tuple.hpp"

namespace RAJA
{
RAJA_HOST_DEVICE
constexpr std::size_t align(std::size_t size, std::size_t alignment = 16)
/** Returns the aligned size. This would use `alignof(std::max_align_t)`;
 *  however, the device side can get a different value compared to the
 *  host.
 *
 * @return The size with proper alignment
 */
{
  return ((size + alignment - 1) / alignment) * alignment;
}

struct msg_header
{
  std::size_t sz;
  int id;
  char* args;
};

template<typename... Args>
using msg_args = camp::tuple<Args...>;
}  // namespace RAJA

#endif  // RAJA_MSG_ALIGN_HPP
