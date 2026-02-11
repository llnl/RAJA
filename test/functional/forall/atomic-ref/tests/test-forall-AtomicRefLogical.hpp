//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for logical atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_LOGICAL_HPP__
#define __TEST_FORALL_ATOMICREF_LOGICAL_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct AndEqOtherOp : int_op {
  AndEqOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max((T)seg.size()),
    final_min(min), final_max(min)
  {
    hcount[0] = np2m1((T)seg.size());
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other &= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchAndOtherOp : int_op {
  FetchAndOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  {
    hcount[0] = max;
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_and((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct OrEqOtherOp : int_op {
  OrEqOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max(np2m1((T)seg.size())),
    final_min(max), final_max(max)
  {
    hcount[0] = T(0);
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other |= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchOrOtherOp : int_op {
  FetchOrOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max(np2m1((T)seg.size())),
    final_min(max), final_max(max)
  {
    hcount[0] = T(0);
    work_res.memcpy(dcount, hcount, sizeof(T));
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_or((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct XorEqOtherOp : int_op {
  XorEqOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  {
    hcount[0] = T(0);
    work_res.memcpy(dcount, hcount, sizeof(T));
    for (IdxType i = 0; i < seg.size(); ++i) {
      final_min ^= (T)i; final_max ^= (T)i;
    }
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other ^= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchXorOtherOp : int_op {
  FetchXorOtherOp(T* dcount, T* hcount, camp::resources::Resource work_res, RAJA::TypedRangeSegment<IdxType> seg)
    : other(dcount), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  {
    hcount[0] = T(0);
    work_res.memcpy(dcount, hcount, sizeof(T));
    for (IdxType i = 0; i < seg.size(); ++i) {
      final_min ^= (T)i; final_max ^= (T)i;
    }
  }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_xor((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template  < typename ExecPolicy,
            typename AtomicPolicy,
            typename IdxType,
            typename T,
            template <typename, typename, typename> class OtherOp>
// No test when underlying op type is int, and index type is float
typename std::enable_if<
           (std::is_floating_point<T>::value && 
            std::is_base_of<int_op, OtherOp<T,AtomicPolicy, IdxType>>::value)
         >::type
testAtomicRefLogicalOp(RAJA::TypedRangeSegment<IdxType> RAJA_UNUSED_ARG(seg), 
                     T* RAJA_UNUSED_ARG(count), T* RAJA_UNUSED_ARG(list),
                     T* RAJA_UNUSED_ARG(hcount), T* RAJA_UNUSED_ARG(hlist),
                     camp::resources::Resource RAJA_UNUSED_ARG(work_res), IdxType RAJA_UNUSED_ARG(N))
{
}

template  < typename ExecPolicy,
            typename AtomicPolicy,
            typename IdxType,
            typename T,
            template <typename, typename, typename> class OtherOp>
// Run test if T is integral and operation is int_op, or for any all_op
typename std::enable_if<
           (std::is_integral<T>::value && 
            std::is_base_of<int_op, OtherOp<T,AtomicPolicy, IdxType>>::value) || 
            (std::is_base_of<all_op, OtherOp<T,AtomicPolicy, IdxType>>::value)
         >::type
testAtomicRefLogicalOp(RAJA::TypedRangeSegment<IdxType> seg, T* count, T* list,
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
void ForallAtomicRefLogicalTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);

  camp::resources::Resource work_res{WorkingRes::get_default()};

  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  T * count   = work_res.allocate<T>(1);
  T * list    = work_res.allocate<T>(N);

  T * hcount   = host_res.allocate<T>(1);
  T * hlist    = host_res.allocate<T>(N);

  // Note: These integral tests require return type conditional overloading 
  //       of testAtomicRefLogicalOp
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       AndEqOtherOp   >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchAndOtherOp>(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       OrEqOtherOp    >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchOrOtherOp >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       XorEqOtherOp   >(seg, count, list, hcount, hlist, work_res, N);
  testAtomicRefLogicalOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchXorOtherOp>(seg, count, list, hcount, hlist, work_res, N);

  work_res.deallocate( count );
  work_res.deallocate( list );
  host_res.deallocate( hcount );
  host_res.deallocate( hlist );
}


TYPED_TEST_SUITE_P(ForallAtomicRefLogicalTest);
template <typename T>
class ForallAtomicRefLogicalTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefLogicalTest, AtomicRefLogicalForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefLogicalTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefLogicalTest,
                            AtomicRefLogicalForall);

#endif  //__TEST_FORALL_ATOMICREF_LOGICAL_HPP__
