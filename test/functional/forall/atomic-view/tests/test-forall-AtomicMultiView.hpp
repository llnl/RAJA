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

#ifndef __TEST_FORALL_ATOMIC_MULTIVIEW_HPP__
#define __TEST_FORALL_ATOMIC_MULTIVIEW_HPP__

#include <cmath>

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename T>
void ForallAtomicMultiViewTestImpl( IdxType N )
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

  // assumes each source[] will be 2x size of each dest[], src_side x dst_side
  RAJA::forall<ExecPolicy>(seg_srcside, [=] RAJA_HOST_DEVICE(IdxType ii)
  {
    source[ii] = actualsource+(ii*dst_side);
  });

  // assumes each dest[] will be a square matrix, dst_side x dst_side
  RAJA::forall<ExecPolicy>(seg_dstside, [=] RAJA_HOST_DEVICE(IdxType ii)
  {
    dest[ii] = actualdest+(ii*dst_side);
  });

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    actualsource[i] = (T)1;
  });

  // use atomic add to reduce the array
  // 1D defaut MultiView
  RAJA::MultiView<T, RAJA::Layout<1>> vec_view(source, N);

  // 1D MultiView with array-of-pointers index in 1st position
  RAJA::MultiView<T, RAJA::Layout<1>, 1> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic MultiView
  RAJA::forall<ExecPolicy>(seg_dstside, [=] RAJA_HOST_DEVICE(IdxType i) {
    for ( int aopidx = 0; aopidx < dst_side; ++aopidx )
    {
      sum_atomic_view(i,aopidx) = (T)0;
    }
  });

  // Assign values to dest using atomic MultiView
  RAJA::forall<ExecPolicy>(seg_srcside, [=] RAJA_HOST_DEVICE(IdxType i) {
    for ( int aopidx = 0; aopidx < dst_side; ++aopidx )
    {
      sum_atomic_view(i/2, aopidx) += vec_view(aopidx,i/2);
    }
  });

  work_res.memcpy( check_array, actualdest, sizeof(T) * N/2 );
  work_res.wait();

  for (IdxType i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, check_array[i]);
  }

  work_res.deallocate( actualsource );
  work_res.deallocate( source );
  work_res.deallocate( actualdest );
  work_res.deallocate( dest );
  host_res.deallocate( check_array );
}

TYPED_TEST_SUITE_P(ForallAtomicMultiViewTest);
template <typename T>
class ForallAtomicMultiViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicMultiViewTest, AtomicMultiViewForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicMultiViewTestImpl<AExec, APol, ResType, IdxType, DType>( 20000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicMultiViewTest,
                            AtomicMultiViewForall);

#endif  //__TEST_FORALL_ATOMIC_MULTIVIEW_HPP__
