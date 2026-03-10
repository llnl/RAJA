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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MSG_HEADER_HPP
#define RAJA_MSG_HEADER_HPP

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
  std::size_t id;
  char* args;
};

template<typename... Args>
struct msg_args
{
  camp::tuple<Args...> args;
};
}  // namespace RAJA

#endif  // RAJA_MSG_HEADER_HPP
