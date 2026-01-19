//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
#define __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__

#if defined(RAJA_ENABLE_SYCL)
#define DEVICE_KERNEL SyclKernel
#elif defined(RAJA_ENABLE_HIP)
#define DEVICE_KERNEL HipKernel
#else
#define DEVICE_KERNEL CudaKernel
#endif

struct DEPTH_1_REDUCESUM {};
struct DEPTH_2 {};
struct DEPTH_2_COLLAPSE {};
struct DEPTH_3 {};
struct DEPTH_3_COLLAPSE {};
struct DEPTH_3_COLLAPSE_SEQ_INNER {};
struct DEPTH_3_COLLAPSE_SEQ_OUTER {};
struct DEPTH_3_REDUCESUM {};
struct DEPTH_3_REDUCESUM_SEQ_INNER {};
struct DEPTH_3_REDUCESUM_SEQ_OUTER {};
struct DEVICE_DEPTH_1_REDUCESUM {};
struct DEVICE_DEPTH_1_REDUCESUM_WARP {};
struct DEVICE_DEPTH_1_REDUCESUM_WARPDIRECT_TILE {};
struct DEVICE_DEPTH_1_REDUCESUM_WARPREDUCE {};
struct DEVICE_DEPTH_2 {};
struct DEVICE_DEPTH_2_REDUCESUM_WARP {};
struct DEVICE_DEPTH_2_REDUCESUM_WARPMASK {};
struct DEVICE_DEPTH_2_REDUCESUM_WARPMASK_FORI {};
struct DEVICE_DEPTH_2_REDUCESUM_WARPREDUCE {};
struct DEVICE_DEPTH_3 {};
struct DEVICE_DEPTH_3_REDUCESUM {};
struct DEVICE_DEPTH_3_REDUCESUM_SEQ_INNER {};
struct DEVICE_DEPTH_3_REDUCESUM_SEQ_OUTER {};
struct DEVICE_DEPTH_3_REDUCESUM_WARPREDUCE {};


//
//
// Nested Loop Data Type information
//
//
template<typename LoopPolType, typename... Policies> 
struct NestedLoopData : camp::list<Policies...> {
  using LoopType = LoopPolType;
};


//
//
// Filter out a list of "NestedLoopData" types given a 
// tests' supported loop Type list.
//
//
namespace detail{

  using namespace camp;

  template<typename T, typename Elements>
  struct is_in_type_list;

  template<typename T, typename Elements>
  struct KELB_impl;

  template<typename T, typename First, typename... Rest>
  struct is_in_type_list<T, list<First, Rest...>> :
    std::conditional<
      std::is_same<  typename T::LoopType, First  >::value,
      list<T>,
      typename is_in_type_list<T, list<Rest...>>::type > {};

  template<typename T, typename Last>
  struct is_in_type_list<T, list<Last>> :
    std::conditional<
      std::is_same< typename T::LoopType , Last>::value,
      list<T>,
      list<> > {};

  template<typename POL_TYPE_LIST, typename First, typename... Rest>
  struct KELB_impl<POL_TYPE_LIST, list<First, Rest...>> :
    join< typename KELB_impl<POL_TYPE_LIST, list<First  >>::type, 
          typename KELB_impl<POL_TYPE_LIST, list<Rest...>>::type > {};

  template<typename POL_TYPE_LIST, typename Last>
  struct KELB_impl<POL_TYPE_LIST, list<Last>> :
    is_in_type_list<Last, POL_TYPE_LIST > {};

} // namespace detail


template<typename POL_TYPE_LIST, typename EXEC_POL_LIST>
struct KernelExecListBuilder {
  using type = typename detail::KELB_impl<POL_TYPE_LIST, EXEC_POL_LIST>::type;
};


#endif  // __TEST_KERNEL_NESTED_LOOP_TYPES_HPP__
