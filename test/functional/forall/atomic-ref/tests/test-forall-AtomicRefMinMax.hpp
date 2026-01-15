//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for min/max atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_MINMAX_HPP__
#define __TEST_FORALL_ATOMICREF_MINMAX_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct MaxEqOtherOp : all_op {
  MaxEqOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  {
    hcount[0] = (T)0;
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchMaxOtherOp : all_op {
  FetchMaxOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  {
    hcount[0] = (T)0;
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct MinEqOtherOp : all_op {
  MinEqOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max((T)seg.size() - (T)1),
    final_min(min), final_max(min)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchMinOtherOp : all_op {
  FetchMinOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max((T)seg.size()),
    final_min(min), final_max(min)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template  < typename ExecPolicy,
            typename AtomicPolicy,
            typename IdxType,
            typename T,
            template <typename, typename, typename> class OtherOp>
void
testAtomicRefMinMaxOp(RAJA::TypedRangeSegment<IdxType> seg, T* count, T* list,
    T* hcount, T* hlist,
    camp::resources::Resource work_res, IdxType N)
{
  OtherOp<T, AtomicPolicy, IdxType> otherop(count, hcount, work_res, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      list[i] = otherop.max + (T)1;
  });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      T val = otherop(i);
      list[i] = val;
  });

  work_res.memcpy( hcount, count, sizeof(T) );
  work_res.memcpy( hlist, list, sizeof(T) * N );
  work_res.wait();

  EXPECT_LE(otherop.final_min, hcount[0]);
  EXPECT_GE(otherop.final_max, hcount[0]);
  for (IdxType i = 0; i < seg.size(); i++) {
    EXPECT_LE(otherop.min, hlist[i]);
    EXPECT_GE(otherop.max, hlist[i]);
  }
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename T>
void ForallAtomicRefMinMaxTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);

  camp::resources::Resource work_res{WorkingRes::get_default()};

  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  T * count   = work_res.allocate<T>(1);
  T * list    = work_res.allocate<T>(N);

  T * hcount   = host_res.allocate<T>(1);
  T * hlist    = host_res.allocate<T>(N);

  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       MaxEqOtherOp   >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchMaxOtherOp>(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       MinEqOtherOp   >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchMinOtherOp>(seg, count, list, hcount, hlist, work_res, N);

  work_res.deallocate( count );
  work_res.deallocate( list );
  host_res.deallocate( hcount );
  host_res.deallocate( hlist );
}


TYPED_TEST_SUITE_P(ForallAtomicRefMinMaxTest);
template <typename T>
class ForallAtomicRefMinMaxTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefMinMaxTest, AtomicRefMinMaxForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefMinMaxTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefMinMaxTest,
                            AtomicRefMinMaxForall);

#endif  //__TEST_FORALL_ATOMICREF_MINMAX_HPP__
