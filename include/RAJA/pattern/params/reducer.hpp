//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef NEW_REDUCE_HPP
#define NEW_REDUCE_HPP

#include <type_traits>

#include "RAJA/pattern/detail/TypeTraits.hpp"
#include "RAJA/pattern/params/params_base.hpp"
#include "RAJA/util/SoAPtr.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#elif defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#elif defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#endif

namespace RAJA
{

namespace operators
{

template<typename T, typename IndexType>
struct limits<RAJA::expt::ValLoc<T, IndexType>>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T, IndexType>
  min()
  {
    return RAJA::expt::ValLoc<T, IndexType>(RAJA::operators::limits<T>::min());
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T, IndexType>
  max()
  {
    return RAJA::expt::ValLoc<T, IndexType>(RAJA::operators::limits<T>::max());
  }
};

}  //  namespace operators

}  //  namespace RAJA

namespace RAJA
{

namespace expt
{
namespace detail
{

#if defined(RAJA_CUDA_ACTIVE)
using device_mem_pool_t = RAJA::cuda::device_mempool_type;
#elif defined(RAJA_HIP_ACTIVE)
using device_mem_pool_t = RAJA::hip::device_mempool_type;
#elif defined(RAJA_SYCL_ACTIVE)
using device_mem_pool_t = RAJA::sycl::device_mempool_type;
#endif

//
//
// Basic Reducer
//
//

// Basic data type Reducer
// T must be a basic data type
// VOp must be ValOp<T, Op>
template<typename Op, typename T, typename VOp>
struct Reducer : public ForallParamBase
{
  using op         = Op;
  using value_type = T;  // This is a basic data type

  Reducer() = default;

  // Basic data type constructor
  RAJA_HOST_DEVICE Reducer(value_type* target_in)
      : m_valop(VOp {}),
        target(target_in)
  {}

  Reducer(Reducer const&)            = default;
  Reducer(Reducer&&)                 = default;
  Reducer& operator=(Reducer const&) = default;
  Reducer& operator=(Reducer&&)      = default;

  // Internal ValOp object that is used within RAJA::forall/launch
  VOp m_valop = VOp {};

  // Points to the user specified result variable
  value_type* target = nullptr;

  // combineTarget() performs the final op on the target data and location in
  // param_resolve()
  RAJA_HOST_DEVICE void combineTarget(value_type in)
  {
    value_type temp = op {}(*target, in);
    *target         = temp;
  }

  RAJA_HOST_DEVICE
  value_type& getVal() { return m_valop.val; }

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) ||                   \
    defined(RAJA_SYCL_ACTIVE)
  // Device related attributes.
  value_type* devicetarget = nullptr;
  RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
  unsigned int* device_count = nullptr;
#endif

  // These are types and parameters extracted from this struct, and given to the
  // forall.
  using ARG_TUP_T = camp::tuple<VOp*>;
  using ARG_T     = VOp;

  RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup()
  {
    return camp::make_tuple(&m_valop);
  }

  RAJA_HOST_DEVICE ARG_T* get_lambda_arg() { return &m_valop; }

  using ARG_LIST_T                        = typename ARG_TUP_T::TList;
  static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
};

// Partial specialization of Reducer for ValLoc
// T is a deduced basic data type
// I is a deduced index type
template<typename T,
         typename I,
         template<typename, typename, typename>
         class Op>
struct Reducer<Op<ValLoc<T, I>, ValLoc<T, I>, ValLoc<T, I>>,
               ValLoc<T, I>,
               ValOp<ValLoc<T, I>, Op>> : public ForallParamBase
{
  using target_value_type = T;
  using target_index_type = I;
  using value_type        = ValLoc<T, I>;
  using op                = Op<value_type, value_type, value_type>;
  using VOp = ValOp<ValLoc<target_value_type, target_index_type>, Op>;

  Reducer() = default;

  // ValLoc constructor
  // Note that the target_ variables point to the val and loc within the user
  // defined target ValLoc
  RAJA_HOST_DEVICE Reducer(value_type* target_in)
      : m_valop(VOp {}),
        target_value(&target_in->val),
        target_index(&target_in->loc)
  {}

  // Dual input constructor for ReduceLoc<>(data, index) case
  // The target_ variables point to vars defined by the user
  RAJA_HOST_DEVICE Reducer(target_value_type* data_in,
                           target_index_type* index_in)
      : m_valop(VOp {}),
        target_value(data_in),
        target_index(index_in)
  {}

  Reducer(Reducer const&)            = default;
  Reducer(Reducer&&)                 = default;
  Reducer& operator=(Reducer const&) = default;
  Reducer& operator=(Reducer&&)      = default;

  // The ValLoc within m_valop is initialized with data and location values from
  // either a ValLoc, or dual data and location values, passed into the
  // constructor
  VOp m_valop = VOp {};

  // Points to either dual value and index defined by the user, or value and
  // index within a ValLoc defined by the user
  target_value_type* target_value = nullptr;
  target_index_type* target_index = nullptr;

  // combineTarget() performs the final op on the target data and location in
  // param_resolve()
  RAJA_HOST_DEVICE void combineTarget(value_type in)
  {
    // Create a different temp ValLoc solely for combining
    value_type temp(*target_value, *target_index);
    temp          = op {}(temp, in);
    *target_value = temp.val;
    *target_index = temp.loc;
  }

  RAJA_HOST_DEVICE
  value_type& getVal() { return m_valop.val; }

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) ||                   \
    defined(RAJA_SYCL_ACTIVE)
  // Device related attributes.
  value_type* devicetarget = nullptr;
  RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
  unsigned int* device_count = nullptr;
#endif

  // These are types and parameters extracted from this struct, and given to the
  // forall.
  using ARG_TUP_T = camp::tuple<VOp*>;
  using ARG_T     = VOp;

  RAJA_HOST_DEVICE ARG_T* get_lambda_arg() { return &m_valop; }

  RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup()
  {
    return camp::make_tuple(&m_valop);
  }

  using ARG_LIST_T                        = typename ARG_TUP_T::TList;
  static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
};

}  // namespace detail

// Standard use case.
template<template<typename, typename, typename> class Op, typename T>
auto constexpr Reduce(T* target)
{
  return detail::Reducer<Op<T, T, T>, T, ValOp<T, Op>>(target);
}

// User-defined ValLoc case.
template<template<typename, typename, typename> class Op,
         typename T,
         typename IndexType>
auto constexpr Reduce(ValLoc<T, IndexType>* target)
{
  using VL = ValLoc<T, IndexType>;
  return detail::Reducer<Op<VL, VL, VL>, VL, ValOp<ValLoc<T, IndexType>, Op>>(
      target);
}

// Dual input use case where reduction value and location are separate,
// non-ValLoc types supplied by the user.
template<template<typename, typename, typename> class Op,
         typename T,
         typename IndexType>
auto constexpr ReduceLoc(T* target, IndexType* index)
{
  using VL = ValLoc<T, IndexType>;
  return detail::Reducer<Op<VL, VL, VL>, VL, ValOp<ValLoc<T, IndexType>, Op>>(
      target, index);
}


}  // namespace expt


}  //  namespace RAJA

#endif  //  NEW_REDUCE_HPP
