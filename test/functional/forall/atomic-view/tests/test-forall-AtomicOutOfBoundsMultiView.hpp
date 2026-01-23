//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
///

#ifndef __TEST_FORALL_ATOMICOUTOFBOUNDS_MULTIVIEW_HPP__
#define __TEST_FORALL_ATOMICOUTOFBOUNDS_MULTIVIEW_HPP__

#include <cmath>

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename T>
void ForallAtomicOutOfBoundsMultiViewTestImpl( IdxType N )
{
  // Functionally similar to ForallAtomicViewTestImpl

  int dst_side = static_cast<int>( std::sqrt( static_cast<double>(N/2) ) ); // dest[] dimension
  int src_side = dst_side*2; // source[] dimension

  RAJA::TypedRangeSegment<IdxType> seg(0, N);
  RAJA::TypedRangeSegment<IdxType> seg_dstside(0, dst_side);
  RAJA::TypedRangeSegment<IdxType> seg_srcside(0, src_side);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  T *  actualsource = work_res.allocate<T> (N);
  T ** source       = work_res.allocate<T*>(src_side);
  T *  actualdest   = work_res.allocate<T> (N/2);
  T ** dest         = work_res.allocate<T*>(dst_side);
  T *  check_array  = host_res.allocate<T> (N/2);

  // PASS_REGEX: Negative index while accessing array of pointers

  // use atomic add to reduce the array
  // 1D defaut MultiView
  RAJA::MultiView<T, RAJA::Layout<1>> vec_view(source, N);

  // 1D MultiView with array-of-pointers index in 1st position
  RAJA::MultiView<T, RAJA::Layout<1>, 1> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Need gtest death test to avoid complete failure due to eventual seg fault
  #if defined(RAJA_ENABLE_TARGET_OPENMP)
  EXPECT_DEATH_IF_SUPPORTED( (sum_atomic_view(0,-1) = (T)0), "" );
  #else
  EXPECT_THROW( (sum_atomic_view(0,-1) = (T)0), std::runtime_error );
  #endif

  work_res.wait();

  work_res.deallocate( actualsource );
  work_res.deallocate( source );
  work_res.deallocate( actualdest );
  work_res.deallocate( dest );
  host_res.deallocate( check_array );
}

TYPED_TEST_SUITE_P(ForallAtomicOutOfBoundsMultiViewTest);
template <typename T>
class ForallAtomicOutOfBoundsMultiViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicOutOfBoundsMultiViewTest, AtomicOutOfBoundsMultiViewForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicOutOfBoundsMultiViewTestImpl<AExec, APol, ResType, IdxType, DType>( 20000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicOutOfBoundsMultiViewTest,
                            AtomicOutOfBoundsMultiViewForall);

#endif  //__TEST_FORALL_ATOMICOUTOFBOUNDS_MULTIVIEW_HPP__
