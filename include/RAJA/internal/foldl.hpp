/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with support for pre-C++14 compilers.
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

#ifndef RAJA_foldl_HPP
#define RAJA_foldl_HPP

#include "RAJA/config.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

// Basics, using c++14 semantics in a c++11 compatible way, credit to libc++

// Forward
namespace detail
{
// FoldL
template<typename Op, typename... Rest>
struct foldl_impl;

template<typename Op, typename Arg1>
struct foldl_impl<Op, Arg1>
{
  using Ret = Arg1;
};

#if RAJA_HAS_CXX17_IS_INVOCABLE

template<typename Op, typename Arg1, typename Arg2>
struct foldl_impl<Op, Arg1, Arg2>
{
  using Ret = typename std::invoke_result<Op, Arg1, Arg2>::type;
};

template<typename Op,
         typename Arg1,
         typename Arg2,
         typename Arg3,
         typename... Rest>
struct foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>
{
  using Ret =
      typename foldl_impl<Op,
                          typename std::invoke_result<
                              Op,
                              typename std::invoke_result<Op, Arg1, Arg2>::type,
                              Arg3>::type,
                          Rest...>::Ret;
};

#else

template<typename Op, typename Arg1, typename Arg2>
struct foldl_impl<Op, Arg1, Arg2>
{
  using Ret = typename std::result_of<Op(Arg1, Arg2)>::type;
};

template<typename Op,
         typename Arg1,
         typename Arg2,
         typename Arg3,
         typename... Rest>
struct foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>
{
  using Ret = typename foldl_impl<
      Op,
      typename std::result_of<Op(typename std::result_of<Op(Arg1, Arg2)>::type,
                                 Arg3)>::type,
      Rest...>::Ret;
};

#endif

}  // namespace detail

template<typename Op, typename Arg1>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(
    Op&& RAJA_UNUSED_ARG(operation),
    Arg1&& arg) -> typename detail::foldl_impl<Op, Arg1>::Ret
{
  return camp::forward<Arg1>(arg);
}

template<typename Op, typename Arg1, typename Arg2>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2) ->
    typename detail::foldl_impl<Op, Arg1, Arg2>::Ret
{
  return camp::forward<Op>(operation)(camp::forward<Arg1>(arg1),
                                      camp::forward<Arg2>(arg2));
}

template<typename Op,
         typename Arg1,
         typename Arg2,
         typename Arg3,
         typename... Rest>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2,
                                                  Arg3&& arg3,
                                                  Rest&&... rest) ->
    typename detail::foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>::Ret
{
  return foldl(camp::forward<Op>(operation),
               camp::forward<Op>(operation)(
                   camp::forward<Op>(operation)(camp::forward<Arg1>(arg1),
                                                camp::forward<Arg2>(arg2)),
                   camp::forward<Arg3>(arg3)),
               camp::forward<Rest>(rest)...);
}

// Convenience folds
template<typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result sum(Args... args)
{
  return foldl(RAJA::operators::plus<Result>(), args...);
}

template<typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result product(Args... args)
{
  return foldl(RAJA::operators::multiplies<Result>(), args...);
}

template<typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result max(Args... args)
{
  return foldl(RAJA::operators::maximum<Result>(), args...);
}

template<typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result min(Args... args)
{
  return foldl(RAJA::operators::minimum<Result>(), args...);
}


}  // namespace RAJA

#endif
