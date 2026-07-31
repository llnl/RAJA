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
struct MsgHeader
{
  std::size_t sz;
  std::size_t type;
  std::size_t hash;
  char* args;
};

template<typename... Args>
struct MsgArgs
{
  camp::tuple<Args...> args;
};
}  // namespace RAJA

#endif  // RAJA_MSG_HEADER_HPP
