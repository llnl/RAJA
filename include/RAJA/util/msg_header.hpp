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
  struct alignas(std::max_align_t) msg_header
  {
    std::size_t sz; 
    int id;
    char *args;
  };

  template <typename... Args>
  struct alignas(std::max_align_t) msg_args
  {
    camp::tuple<Args...> args;
  };
} // end of RAJA namespace

#endif // RAJA_MSG_ALIGN_HPP
