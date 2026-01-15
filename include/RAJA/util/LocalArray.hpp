/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for multi-dimensional shared memory tile Views.
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


#ifndef RAJA_util_LocalArray_HPP
#define RAJA_util_LocalArray_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/TypedViewBase.hpp"

namespace RAJA
{


template<camp::idx_t... Sizes>
using ParamList = camp::idx_seq<Sizes...>;

/*!
 * RAJA local array
 * Holds a pointer and information necessary
 * to allocate a static array.
 *
 * Once intialized they can be treated as an N dimensional array
 * on the CPU stack, CUDA thread private memory,
 * or CUDA shared memory. Intialization occurs within
 * the RAJA::Kernel statement ``InitLocalArray"
 *
 * An accessor is provided to enable multi-dimensional indexing.
 * Two versions are created below, a strongly typed version and
 * a non-strongly typed version.
 */


namespace internal
{


template<typename Perm, typename Sizes>
struct StaticLayoutHelper;

template<camp::idx_t... Perm, Index_type... Sizes>
struct StaticLayoutHelper<camp::idx_seq<Perm...>, SizeList<Sizes...>>
{
  using type = StaticLayout<camp::idx_seq<Perm...>, Sizes...>;
};

template<typename Perm, typename Sizes>
using getStaticLayoutType = typename StaticLayoutHelper<Perm, Sizes>::type;


}  // namespace internal

template<typename ValueType,
         typename Perm,
         typename Sizes,
         typename... IndexTypes>
using TypedLocalArray =
    internal::TypedViewBase<ValueType,
                            ValueType*,
                            internal::getStaticLayoutType<Perm, Sizes>,
                            camp::list<IndexTypes...>>;


template<typename ValueType, typename Perm, typename Sizes>
using LocalArray =
    internal::TypedViewBase<ValueType,
                            ValueType*,
                            internal::getStaticLayoutType<Perm, Sizes>,
                            internal::getDefaultIndexTypes<Perm>>;

template<typename AtomicPolicy,
         typename DataType,
         typename Perm,
         typename Sizes,
         typename... IndexTypes>
struct AtomicTypedLocalArray
{};

template<typename AtomicPolicy,
         typename DataType,
         camp::idx_t... Perm,
         Index_type... Sizes,
         typename... IndexTypes>
struct AtomicTypedLocalArray<AtomicPolicy,
                             DataType,
                             camp::idx_seq<Perm...>,
                             RAJA::SizeList<Sizes...>,
                             IndexTypes...>
{
  DataType* m_arrayPtr = nullptr;
  using value_type     = DataType;
  using atomic_ref_t   = RAJA::AtomicRef<value_type, AtomicPolicy>;
  using layout_type    = RAJA::StaticLayout<camp::idx_seq<Perm...>, Sizes...>;
  static const camp::idx_t NumElem = layout_type::s_size;

  RAJA_HOST_DEVICE
  atomic_ref_t operator()(IndexTypes... indices) const
  {
    return (atomic_ref_t(
        &m_arrayPtr[layout_type::s_oper(stripIndexType(indices)...)]));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr camp::idx_t size() const { return layout_type::s_size; }

  RAJA_HOST_DEVICE
  RAJA_INLINE void set_data(DataType* data_ptr) { m_arrayPtr = data_ptr; }
};


}  // end namespace RAJA


#endif
