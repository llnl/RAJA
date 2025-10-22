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

#ifndef RAJA_SOA_PTR_HPP
#define RAJA_SOA_PTR_HPP

#include "RAJA/config.hpp"

#include <type_traits>

// for RAJA::reduce::detail::ValueLoc
#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/util/types.hpp"

// for RAJA::expt::ValLoc
#include "RAJA/pattern/params/params_base.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * @brief Pointer class specialized for Struct of Array data layout allocated
 *        via RAJA basic_mempools.
 *
 * This is useful for creating a vectorizable data layout and getting
 * coalesced memory accesses or avoiding shared memory bank conflicts in cuda.
 */
template<typename T,
         typename mempool = RAJA::basic_mempool::MemPool<
             RAJA::basic_mempool::generic_allocator>,
         typename accessor = DefaultAccessor>
class SoAPtr
{
  template<typename, typename, typename>
  friend class SoAPtr;  // friend other instantiations of this class

public:
  using value_type = T;

  template<typename rhs_accessor>
  using rebind_accessor = SoAPtr<T, mempool, rhs_accessor>;

  SoAPtr()                         = default;
  SoAPtr(SoAPtr const&)            = default;
  SoAPtr(SoAPtr&&)                 = default;
  SoAPtr& operator=(SoAPtr const&) = default;
  SoAPtr& operator=(SoAPtr&&)      = default;

  explicit SoAPtr(size_t size)
      : mem(mempool::getInstance().template malloc<value_type>(size))
  {}

  template<
      typename rhs_accessor,
      std::enable_if_t<!std::is_same<accessor, rhs_accessor>::value>* = nullptr>
  RAJA_HOST_DEVICE explicit SoAPtr(
      SoAPtr<value_type, mempool, rhs_accessor> const& rhs)
      : mem(rhs.mem)
  {}

  SoAPtr& allocate(size_t size)
  {
    mem = mempool::getInstance().template malloc<value_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  constexpr RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return accessor::get(mem, i);
  }

  constexpr RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    accessor::set(mem, i, val);
  }

private:
  value_type* mem = nullptr;
};

/*!
 * @brief Specialization for RAJA::reduce::detail::ValueLoc.
 */
template<typename T,
         typename IndexType,
         bool doing_min,
         typename mempool,
         typename accessor>
class SoAPtr<RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>,
             mempool,
             accessor>
{
  using first_type  = T;
  using second_type = IndexType;

  template<typename, typename, typename>
  friend class SoAPtr;  // fiend other instantiations of this class

public:
  using value_type = RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>;

  template<typename rhs_accessor>
  using rebind_accessor = SoAPtr<value_type, mempool, rhs_accessor>;

  SoAPtr()                         = default;
  SoAPtr(SoAPtr const&)            = default;
  SoAPtr(SoAPtr&&)                 = default;
  SoAPtr& operator=(SoAPtr const&) = default;
  SoAPtr& operator=(SoAPtr&&)      = default;

  explicit SoAPtr(size_t size)
      : mem(mempool::getInstance().template malloc<first_type>(size)),
        mem_idx(mempool::getInstance().template malloc<second_type>(size))
  {}

  template<
      typename rhs_accessor,
      std::enable_if_t<!std::is_same<accessor, rhs_accessor>::value>* = nullptr>
  RAJA_HOST_DEVICE explicit SoAPtr(
      SoAPtr<value_type, mempool, rhs_accessor> const& rhs)
      : mem(rhs.mem),
        mem_idx(rhs.mem_idx)
  {}

  SoAPtr& allocate(size_t size)
  {
    mem     = mempool::getInstance().template malloc<first_type>(size);
    mem_idx = mempool::getInstance().template malloc<second_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    mempool::getInstance().free(mem_idx);
    mem_idx = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  constexpr RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return value_type(accessor::get(mem, i), accessor::get(mem_idx, i));
  }

  constexpr RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    accessor::set(mem, i, first_type(val));
    accessor::set(mem_idx, i, val.getLoc());
  }

private:
  first_type* mem      = nullptr;
  second_type* mem_idx = nullptr;
};

/*!
 * @brief Specialization for RAJA::expt::ValLoc.
 */
template<typename T, typename IndexType, typename mempool, typename accessor>
class SoAPtr<RAJA::expt::ValLoc<T, IndexType>, mempool, accessor>
{
  using first_type  = T;
  using second_type = IndexType;

  template<typename, typename, typename>
  friend class SoAPtr;  // friend other instantiations of this class

public:
  using value_type = RAJA::expt::ValLoc<T, IndexType>;

  template<typename rhs_accessor>
  using rebind_accessor = SoAPtr<value_type, mempool, rhs_accessor>;

  SoAPtr()                         = default;
  SoAPtr(SoAPtr const&)            = default;
  SoAPtr(SoAPtr&&)                 = default;
  SoAPtr& operator=(SoAPtr const&) = default;
  SoAPtr& operator=(SoAPtr&&)      = default;

  explicit SoAPtr(size_t size)
      : mem(mempool::getInstance().template malloc<first_type>(size)),
        mem_idx(mempool::getInstance().template malloc<second_type>(size))
  {}

  template<
      typename rhs_accessor,
      std::enable_if_t<!std::is_same<accessor, rhs_accessor>::value>* = nullptr>
  RAJA_HOST_DEVICE explicit SoAPtr(
      SoAPtr<value_type, mempool, rhs_accessor> const& rhs)
      : mem(rhs.mem),
        mem_idx(rhs.mem_idx)
  {}

  SoAPtr& allocate(size_t size)
  {
    mem     = mempool::getInstance().template malloc<first_type>(size);
    mem_idx = mempool::getInstance().template malloc<second_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    mempool::getInstance().free(mem_idx);
    mem_idx = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  constexpr RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return value_type(accessor::get(mem, i), accessor::get(mem_idx, i));
  }

  constexpr RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    accessor::set(mem, i, val.getVal());
    accessor::set(mem_idx, i, val.getLoc());
  }

private:
  first_type* mem      = nullptr;
  second_type* mem_idx = nullptr;
};

}  // namespace detail

}  // namespace RAJA

#endif /* RAJA_SOA_PTR_HPP */
