/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in HIP operations.
 *
 *          These methods work only on platforms that support HIP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hiperrchk_HPP
#define RAJA_hiperrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iostream>
#include <utility>
#include <algorithm>
#include <tuple>
#include <array>
#include <string_view>
#include <string>
#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "camp/defines.hpp"
#include "camp/helpers.hpp"

#include "RAJA/util/macros.hpp"

#if defined(__HIPCC__)
// Tell rocprim to provide its HIP API
#define ROCPRIM_HIP_API 1
#include "rocprim/types.hpp"
#endif

namespace camp
{

namespace experimental
{

template<>
struct StreamInsertHelper<RAJA_HIP_DIM_T&>
{
  RAJA_HIP_DIM_T& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x << "," << m_val.y << "," << m_val.z << "}";
  }
};

///
template<>
struct StreamInsertHelper<RAJA_HIP_DIM_T const&>
{
  RAJA_HIP_DIM_T const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.x << "," << m_val.y << "," << m_val.z << "}";
  }
};

#if defined(__HIPCC__)
template<typename R>
struct StreamInsertHelper<::rocprim::double_buffer<R>&>
{
  ::rocprim::double_buffer<R>& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.current() << "," << m_val.alternate() << "}";
  }
};

///
template<typename R>
struct StreamInsertHelper<::rocprim::double_buffer<R> const&>
{
  ::rocprim::double_buffer<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "{" << m_val.current() << "," << m_val.alternate() << "}";
  }
};
#endif

}  // namespace experimental

}  // namespace camp

namespace RAJA
{

///
///////////////////////////////////////////////////////////////////////
///
/// DEPRECATED
/// Utility assert method used in HIP operations to report HIP
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define hipErrchk(ans)                                                         \
  {                                                                            \
    ::RAJA::hipAssert((ans), __FILE__, __LINE__);                              \
  }

[[deprecated]] inline void hipAssert(hipError_t code,
                                     const char* file,
                                     int line,
                                     bool abort = true)
{
  if (code != hipSuccess)
  {
    if (abort)
    {
      std::string msg;
      msg += "HIPassert: ";
      msg += hipGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    }
    else
    {
      fprintf(stderr, "HIPassert: %s %s %d\n", hipGetErrorString(code), file,
              line);
    }
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
