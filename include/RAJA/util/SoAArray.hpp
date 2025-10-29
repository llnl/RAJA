/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for common RAJA internal definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SOA_ARRAY_HPP
#define RAJA_SOA_ARRAY_HPP

#include "RAJA/config.hpp"

// for RAJA::reduce::detail::ValueLoc
#include "RAJA/pattern/detail/reduce.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * @brief Array class specialized for Struct of Array data layout.
 *
 * This is useful for creating a vectorizable data layout and getting
 * coalesced memory accesses or avoiding shared memory bank conflicts in cuda.
 */
template<typename T, size_t size>
class SoAArray
{
  using value_type = T;

public:
  constexpr RAJA_HOST_DEVICE value_type get(size_t i) const { return mem[i]; }

  constexpr RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    mem[i] = val;
  }

private:
  value_type mem[size];
};

/*!
 * @brief Specialization for RAJA::reduce::detail::ValueLoc.
 */
template<typename T, typename IndexType, bool doing_min, size_t size>
class SoAArray<::RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>, size>
{
  using value_type  = ::RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>;
  using first_type  = T;
  using second_type = IndexType;

public:
  constexpr RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return value_type(mem[i], mem_idx[i]);
  }

  constexpr RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    mem[i]     = val;
    mem_idx[i] = val.getLoc();
  }

private:
  first_type mem[size];
  second_type mem_idx[size];
};

}  // namespace detail

}  // namespace RAJA

#endif /* RAJA_SOA_ARRAY_HPP */
