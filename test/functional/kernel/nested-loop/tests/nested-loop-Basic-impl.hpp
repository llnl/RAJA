//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __NESTED_LOOP_BASIC_IMPL_HPP__
#define __NESTED_LOOP_BASIC_IMPL_HPP__

#include <numeric>

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, WORKING_RES work_res, Args&&... args) {
  RAJA::kernel_resource<EXEC_POL>( segs, work_res, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< !USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, WORKING_RES, Args&&... args) {
  RAJA::kernel<EXEC_POL>( segs, args...);
}

//
//
// Define list of nested loop types the Basic test supports.
//
//
using BasicSupportedLoopTypeList = camp::list<
  DEPTH_2,
  DEPTH_2_COLLAPSE,
  DEPTH_3,
  DEPTH_3_COLLAPSE,
  DEPTH_3_COLLAPSE_SEQ_INNER,
  DEPTH_3_COLLAPSE_SEQ_OUTER,
  DEVICE_DEPTH_2>;

//
//
// Basic 2D Matrix index calculation per element.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... ExtraArgs>
void KernelNestedLoopTest(const DEPTH_2&,
                          const RAJA::Index_type dim0,
                          const RAJA::Index_type dim1,
                          ExtraArgs...){
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type flatSize = dim0 * dim1;
  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(flatSize,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
  RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
  RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);

  std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

  constexpr int Depth = 2;
  RAJA::View< RAJA::Index_type, RAJA::Layout<Depth> > work_view(work_array, dim1, dim0);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(RAJA::make_tuple(range1, range0), work_res,
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type j, RAJA::Index_type i) {
                              work_view(j,i) = (j * dim0) + i;
                            });

  work_res.memcpy(check_array, work_array, sizeof(RAJA::Index_type) * RAJA::stripIndexType(flatSize));
  RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (RAJA::Index_type i) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  });

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

// DEPTH_2_COLLAPSE and DEVICE_DEPTH_2 execution policies use the above DEPTH_2 test.
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_2_COLLAPSE&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RESOURCE>(DEPTH_2(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEVICE_DEPTH_2&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RESOURCE>(DEPTH_2(), args...);
}

//
//
// Basic 3D Matrix index calculation per element.
//
//
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE>
void KernelNestedLoopTest(const DEPTH_3&,
                          const RAJA::Index_type dim0,
                          const RAJA::Index_type dim1,
                          const RAJA::Index_type dim2){
  WORKING_RES work_res{WORKING_RES::get_default()};
  camp::resources::Resource erased_work_res{work_res};

  RAJA::Index_type flatSize = dim0 * dim1 * dim2;
  RAJA::Index_type* work_array;
  RAJA::Index_type* check_array;
  RAJA::Index_type* test_array;

  allocateForallTestData<RAJA::Index_type>(flatSize,
                                     erased_work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
  RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
  RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);
  RAJA::TypedRangeSegment<RAJA::Index_type> range2(0, dim2);

  std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

  constexpr int Depth = 3;
  RAJA::View< RAJA::Index_type, RAJA::Layout<Depth> > work_view(work_array, dim2, dim1, dim0);

  call_kernel<EXEC_POLICY, USE_RESOURCE>(RAJA::make_tuple(range2, range1, range0), work_res,
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type k, RAJA::Index_type j, RAJA::Index_type i) {
                              work_view(k,j,i) = (dim0 * dim1 * k) + (dim0 * j) + i;
                            });

  work_res.memcpy(check_array, work_array, sizeof(RAJA::Index_type) * RAJA::stripIndexType(flatSize));
  RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (RAJA::Index_type i) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  });

  deallocateForallTestData<RAJA::Index_type>(erased_work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

// DEPTH_3_COLLAPSE execution policies use the above DEPTH_3 test.
template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_3_COLLAPSE&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RESOURCE>(DEPTH_3(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_3_COLLAPSE_SEQ_OUTER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RESOURCE>(DEPTH_3(), args...);
}

template <typename WORKING_RES, typename EXEC_POLICY, bool USE_RESOURCE, typename... Args>
void KernelNestedLoopTest(const DEPTH_3_COLLAPSE_SEQ_INNER&, Args... args){
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY, USE_RESOURCE>(DEPTH_3(), args...);
}

//
//
// Defining the Kernel Loop structure for Basic Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename POLICY_DATA>
struct BasicNestedLoopExec;

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_3, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
};

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<0>
        >
      >
    >;
};

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_2_COLLAPSE, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,0>,
        RAJA::statement::Lambda<0>
      >
    >;
};

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_3_COLLAPSE, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<0,1,2>,
        RAJA::statement::Lambda<0>
      >
    >;
};

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_3_COLLAPSE_SEQ_OUTER, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,2>,
          RAJA::statement::Lambda<0>
        >
      >
    >;
};

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEPTH_3_COLLAPSE_SEQ_INNER, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<0,1>,
        RAJA::statement::For<2, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP) or defined(RAJA_ENABLE_SYCL)

template<typename POLICY_DATA>
struct BasicNestedLoopExec<DEVICE_DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<0>>::type,  // row
          RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<1>>::type,  // col
            RAJA::statement::Lambda<0>
          >
        >
      > // end CudaKernel
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP

#endif  // __NESTED_LOOP_BASIC_IMPL_HPP__
