//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for subtraction arithmetic atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_SUB_HPP__
#define __TEST_FORALL_ATOMICREF_SUB_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct PreDecCountOp {
  PreDecCountOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(dcount), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (--counter);
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct PostDecCountOp {
  PostDecCountOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(dcount), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (counter--) - (T)1;
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct SubEqCountOp {
  SubEqCountOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(dcount), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (counter -= (T)1);
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchSubCountOp {
  FetchSubCountOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(dcount), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  {
    hcount[0] = (T)seg.size();
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return counter.fetch_sub((T)1) - (T)1;
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template <typename ExecPolicy,
         typename AtomicPolicy,
         typename IdxType,
         typename T,
         template <typename, typename, typename> class CountOp>
void testAtomicRefSub(RAJA::TypedRangeSegment<IdxType> seg,
    T* count, T* list, bool* hit,
    T* hcount, T* hlist, bool* hhit,
    camp::resources::Resource work_res, IdxType N)
{
  CountOp<T, AtomicPolicy, IdxType> countop(count, hcount, work_res, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      list[i] = countop.max + (T)1;
      hit[i] = false;
      });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      T val = countop(i);
      list[i] = val;
      hit[(IdxType)val] = true;
      });

  work_res.memcpy( hcount, count, sizeof(T) );
  work_res.memcpy( hlist, list, sizeof(T) * N );
  work_res.memcpy( hhit, hit, sizeof(bool) * N );
  work_res.wait();

  EXPECT_EQ(countop.final, hcount[0]);
  for (IdxType i = 0; i < seg.size(); i++) {
    EXPECT_LE(countop.min, hlist[i]);
    EXPECT_GE(countop.max, hlist[i]);
    EXPECT_TRUE(hhit[i]);
  }
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WorkingRes,
          typename IdxType,
          typename T>
void ForallAtomicRefSubTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);

  camp::resources::Resource work_res{WorkingRes::get_default()};

  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  T * count   = work_res.allocate<T>(1);
  T * list    = work_res.allocate<T>(N);
  bool * hit  = work_res.allocate<bool>(N);

  T * hcount   = host_res.allocate<T>(1);
  T * hlist    = host_res.allocate<T>(N);
  bool * hhit  = host_res.allocate<bool>(N);

  testAtomicRefSub<ExecPolicy, AtomicPolicy, IdxType, T, 
                     PreDecCountOp  >(seg, count, list, hit, hcount, hlist, hhit, work_res, N);
  testAtomicRefSub<ExecPolicy, AtomicPolicy, IdxType, T, 
                     PostDecCountOp >(seg, count, list, hit, hcount, hlist, hhit, work_res, N);
  testAtomicRefSub<ExecPolicy, AtomicPolicy, IdxType, T, 
                     SubEqCountOp   >(seg, count, list, hit, hcount, hlist, hhit, work_res, N);
  testAtomicRefSub<ExecPolicy, AtomicPolicy, IdxType, T, 
                     FetchSubCountOp>(seg, count, list, hit, hcount, hlist, hhit, work_res, N);

  work_res.deallocate( count );
  work_res.deallocate( list );
  work_res.deallocate( hit );
  host_res.deallocate( hcount );
  host_res.deallocate( hlist );
  host_res.deallocate( hhit ); 
}


TYPED_TEST_SUITE_P(ForallAtomicRefSubTest);
template <typename T>
class ForallAtomicRefSubTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefSubTest, AtomicRefSubForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefSubTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefSubTest,
                            AtomicRefSubForall);

#endif  //__TEST_FORALL_ATOMICREF_SUB_HPP__
