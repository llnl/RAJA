/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in CUDA operations.
 *
 *          These methods work only on platforms that support CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cudaerrchk_HPP
#define RAJA_cudaerrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iostream>
#include <utility>
#include <algorithm>
#include <tuple>
#include <array>
#include <string_view>
#include <string>
#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "camp/defines.hpp"
#include "camp/helpers.hpp"

#include "RAJA/util/macros.hpp"

#include "cub/util_type.cuh"

namespace camp
{

namespace experimental
{

template<>
struct StreamInsertHelper<RAJA_CUDA_DIM_T&>
{
  RAJA_CUDA_DIM_T& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x << "," << m_val.y << "," << m_val.z << "}";
  }
};

///
template<>
struct StreamInsertHelper<RAJA_CUDA_DIM_T const&>
{
  RAJA_CUDA_DIM_T const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x << "," << m_val.y << "," << m_val.z << "}";
  }
};

template<typename R>
struct StreamInsertHelper<::cub::DoubleBuffer<R>&>
{
  ::cub::DoubleBuffer<R>& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.Current() << "," << m_val.Alternate() << "}";
  }
};

///
template<typename R>
struct StreamInsertHelper<::cub::DoubleBuffer<R> const&>
{
  ::cub::DoubleBuffer<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    // Can't get current and alternate as they are non-const functions
    return str << "{?,?}";
  }
};

}  // namespace experimental

}  // namespace camp

namespace RAJA
{

///
///////////////////////////////////////////////////////////////////////
///
/// DEPRECATED
/// Utility assert method used in CUDA operations to report CUDA
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define cudaErrchk(ans)                                                        \
  {                                                                            \
    ::RAJA::cudaAssert((ans), __FILE__, __LINE__);                             \
  }

[[deprecated]] inline void cudaAssert(cudaError_t code,
                                      const char* file,
                                      int line,
                                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    if (abort)
    {
      std::string msg;
      msg += "CUDAassert: ";
      msg += cudaGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    }
    else
    {
      fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file,
              line);
    }
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
