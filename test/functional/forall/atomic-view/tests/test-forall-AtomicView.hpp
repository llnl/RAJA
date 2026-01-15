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

#ifndef __TEST_FORALL_ATOMIC_VIEW_HPP__
#define __TEST_FORALL_ATOMIC_VIEW_HPP__

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename T>
void ForallAtomicViewTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);
  RAJA::TypedRangeSegment<IdxType> seg_half(0, N / 2);

  camp::resources::Resource work_res{WorkingRes::get_default()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  T * hsource = host_res.allocate<T>(N);
  T * source = work_res.allocate<T>(N);
  T * dest = work_res.allocate<T>(N/2);
  T * check_array = host_res.allocate<T>(N/2);

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](IdxType i) { hsource[i] = (T)1; });

  work_res.memcpy( source, hsource, sizeof(T) * N );

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE(IdxType i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

  work_res.memcpy( check_array, dest, sizeof(T) * N/2 );
  work_res.wait();

  for (IdxType i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, check_array[i]);
  }

  host_res.deallocate( hsource );
  work_res.deallocate( source );
  work_res.deallocate( dest );
  host_res.deallocate( check_array );
}

TYPED_TEST_SUITE_P(ForallAtomicViewTest);
template <typename T>
class ForallAtomicViewTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicViewTest, AtomicViewForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicViewTestImpl<AExec, APol, ResType, IdxType, DType>( 100000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicViewTest,
                            AtomicViewForall);

#endif  //__TEST_FORALL_ATOMIC_VIEW_HPP__
